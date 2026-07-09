### A Pluto.jl notebook ###
# v0.1.0

using Markdown
using InteractiveUtils

# ╔═╡ aaaa0002-0000-4000-8000-000000000002
md"""
# Water–Lipid–Protein Material Decomposition on a QRM-Thorax — a pure-physics study

Recover the **volumetric fractions** ``(f_w, f_l, f_p)`` of water/lipid/protein mixtures from dual-energy CT
on a **stadium QRM-thorax phantom** (faithful PCATSim geometry: two lungs split by a mediastinal muscle column,
ribs, spine, and a heart cavity holding the material inserts). Everything is inline (the only data file is the
Woodard adipose CSV for the prior): mix materials by volume fraction, simulate 80/140-kVp DECT with
[BasisSimulator.jl](https://github.com/MolloiLab/BasisSimulator.jl) **(v0.8.0, `:dd_fast`)**, synthesize VMI at
**70 and 150 keV** (the `WLP_PAIR` knob; 150 keV is a clinically standard VMI), and invert.

**Two complementary products.**
1. A **quadratic calibration surface** ``f=\\mathrm{poly}_2(\\mathrm{HU}_{40},\\mathrm{HU}_{70})`` for per-voxel
   point accuracy, delivered as a **boundary-agnostic** map (per-voxel decode + σ_f-weighted edge-preserving
   Huber-TV — never the ground-truth boundary, which real fat doesn't give you).
2. An **integrated-HU** (mass-conservation) estimator for the PVE-robust *total* lipid: a normalized recon PSF
   conserves the integral, so integrating an **affine** decode of the linearly-mixing HU over a generous region
   with a **local muscle background** recovers the lipid that partial volume spreads past the visible edge — the
   quantity the object-extent measure under-reports for small fat.

**Validation:** calibrate on packed circular inserts; test on **held-out circular + sector** geometry.

!!! warning "Requires BasisSimulator v0.8.0 (`:dd_fast`) and a wide, bowtie-free scan geometry"
    The recon must be quantitative (pure lipid −204 vs theoretical −213, <5%). The 350 mm fat ring also needs a
    **wide detector (1300 cols ≈ 415 mm scan FOV) and `bowtie=:none`** or the ring truncates/fades; and **984
    views** to suppress aliasing. Validate the sim against pure-material HU before trusting any decode.
"""

# ╔═╡ aaaa0003-0000-4000-8000-000000000003
md"## 1 · Setup"

# ╔═╡ aaaa0001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(@__DIR__)
end

# ╔═╡ aaaa0004-0000-4000-8000-000000000004
begin
    import BasisSimulator as BS
    import Metal
    import CairoMakie as CM
    using Unitful, LinearAlgebra, Statistics, Random, DelimitedFiles, Printf, Serialization
    to_gpu(x) = Metal.functional() ? Metal.MtlArray(x) : x
    const DATA = joinpath(@__DIR__, "data")
    const ASSET = joinpath(@__DIR__, "assets")
    safe_save(p, f; pu=1.4) = CM.save(p, f; px_per_unit=pu)   # figures ≤1520 wide ⇒ ≤2130 px; keep ≤2000 side
    md"imports · GPU backend (Metal, CPU fallback) · `safe_save`"
end

# ╔═╡ aaaa0005-0000-4000-8000-000000000005
md"## 2 · Materials & theoretical endpoints"

# ╔═╡ aaaa0006-0000-4000-8000-000000000006
begin
    const PROTEIN = BS.XA.Material("protein_WW1986", 0.0, 0.0u"eV", 1.35u"g/cm^3",
        Dict{Int,Float64}(1=>0.066, 6=>0.534, 7=>0.170, 8=>0.220, 16=>0.010))   # Woodard & White 1986
    const WATER = BS.XA.Materials.basis_water; const LIPID = BS.XA.Materials.basis_lipid
    ρval(m) = BS.XA.val(m.density); const ΡW, ΡL, ΡP = ρval(WATER), ρval(LIPID), ρval(PROTEIN)
    function wlp_material(fw, fl, fp; name="wlp")
        ρ = fw*ΡW + fl*ΡL + fp*ΡP; mf = (fw*ΡW/ρ, fl*ΡL/ρ, fp*ΡP/ρ); comp = Dict{Int,Float64}()
        for (m,wm) in ((WATER,mf[1]),(LIPID,mf[2]),(PROTEIN,mf[3])), (Z,f) in m.composition
            comp[Z] = get(comp,Z,0.0) + wm*f
        end
        BS.XA.Material(name, 0.0, 0.0u"eV", ρ*u"g/cm^3", comp)
    end
    theo_hu(mat, E) = (μw = BS.compute_μ_at_energy(WATER, E); 1000.0*(BS.compute_μ_at_energy(mat, E) - μw)/μw)
    mix_hu(fw, fl, fp, E) = theo_hu(wlp_material(fw,fl,fp), E)
    const WLP_PAIR = (70.0, 150.0)        # ← the keV pair knob (this branch tests 70/150; main uses 40/70)
    const E40, E70 = WLP_PAIR             # historical slot names: E40=low, E70=high energy of the pair
    const PL = (theo_hu(LIPID,E40), theo_hu(LIPID,E70)); const PP = (theo_hu(PROTEIN,E40), theo_hu(PROTEIN,E70)); const PW = (0.0, 0.0)
    md"Endpoints (theoretical HU): water=(0,0), lipid=$(round.(PL,digits=0)), protein=$(round.(PP,digits=0))."
end

# ╔═╡ aaaa0007-0000-4000-8000-000000000007
md"## 3 · Composition generation"

# ╔═╡ aaaa0008-0000-4000-8000-000000000008
begin
    const ADI = ("healthy","obese","reduced","comparison","unspecified")
    mass_to_vol(w,l,p) = (v=(w/ΡW,l/ΡL,p/ΡP); s=sum(v); v./s)
    function draw_wlp(seed, n; csv=joinpath(DATA,"adipose_composition_distribution.csv"))
        raw=readdlm(csv,','; header=false); hdr=string.(raw[1,:]); rows=raw[2:end,:]
        ci(x)=findfirst(==(x),hdr); cc,cl,cp,cs=ci("component"),ci("lipid_pct"),ci("component_pct"),ci("state")
        keep(i)=string(rows[i,cs]) in ADI && Float64(rows[i,cl])≥50
        L=[Float64(rows[i,cl]) for i in axes(rows,1) if keep(i)]; sl=Float64[];sp=Float64[]
        for i in axes(rows,1);(keep(i)&&string(rows[i,cc])=="protein")||continue;push!(sl,Float64(rows[i,cl]));push!(sp,Float64(rows[i,cp]));end
        rng=MersenneTwister(seed); bwL=std(L)*length(L)^(-1/5); q=sp./(100 .-sl); z=log.(q./(1 .-q)); μz,σz=mean(z),std(z)
        [(l=clamp(L[rand(rng,1:length(L))]+bwL*randn(rng),50.,99.);qq=1/(1+exp(-(μz+σz*randn(rng))));p=qq*(100-l);mass_to_vol(100-l-p,l,p)) for _ in 1:n]
    end
    function diverse_comps(seed, n)
        rng=MersenneTwister(seed); out=NTuple{3,Float64}[]
        for _ in 1:n; fl=0.05+0.9*rand(rng);fp=0.3*rand(rng);fw=1-fl-fp; fw<0.02&&(fl-=0.02-fw;fw=0.02); push!(out,(fw,fl,fp)); end
        out
    end
    md"`draw_wlp` (adipose prior, Woodard KDE) · `diverse_comps` (spanning test set)"
end

# ╔═╡ aaaa0009-0000-4000-8000-000000000009
md"""## 4 · Stadium QRM-thorax phantom & forward acquisition
Faithful PCATSim geometry: fat-ring/muscle/lung **stadiums**, two lungs split by the mediastinal muscle
bridges, ribs, spine with vertebral arch, and a **heart cavity** holding the inserts. Heart holds **13
hex-packed ø19.9 mm circular inserts** (insert size *derived* from the recon pixel so an eroded core ≥ 225 mm²),
or **8×2 solid sectors** (held-out shape), or a single centred insert (integrated-HU size series).
Dual-kVp EICT (`:dd_fast`) → Cong water/iodine basis → FBP → VMI at 70 & 150 keV (`WLP_PAIR`)."""

# ╔═╡ aaaa0010-0000-4000-8000-000000000010
begin
    const ROD0 = 8; const VOXMM = 0.4                              # first insert label · phantom voxel (mm)
    const RECON_FOV_MM = 380.0; const RECON_N = 512               # recon geometry — SSoT (fov_cm × 10)
    const RECON_PX_MM  = RECON_FOV_MM / RECON_N                    # 0.7422 mm/recon-px (ROIs live on the recon grid)
    const HC_X, HC_Y = 185.0, 115.0                                # heart cavity centre (image-frame mm)
    const HEART_R_MM = 55.0
    const PACK_R_MM = 50.0; const EROSION_PX = 2; const MIN_AREA_MM2 = 225.0; const INSERT_GAP_MM = 2.5
    insert_radius_for_area(min_area, erosion_px, px=RECON_PX_MM) = sqrt(min_area/π) + erosion_px*px
    function pack_inserts(container_r, insert_r, gap)             # hex-lattice fill; count DERIVED from pixel size
        step=2insert_r+gap; h=step*sqrt(3)/2; Rc=container_r-insert_r; cs=NTuple{2,Float64}[]; nrow=ceil(Int,Rc/h)+1
        for row in -nrow:nrow; y=row*h; xoff=isodd(row) ? step/2 : 0.0
            for col in -(ceil(Int,Rc/step)+2):(ceil(Int,Rc/step)+2); x=col*step+xoff; hypot(x,y)≤Rc+1e-9 && push!(cs,(x,y)); end; end
        cs
    end
    const INS_R  = insert_radius_for_area(MIN_AREA_MM2, EROSION_PX)   # ø19.9 mm
    const HEARTC = pack_inserts(PACK_R_MM, INS_R, INSERT_GAP_MM)      # 13 packed insert centres (mm, rel heart)
    const NHEART = length(HEARTC)
    const SPARSE_RADII = [4.0,6.0,9.0,12.0]                           # integrated-HU size series (centred, one per sim)
    const SECT_R_MM = 50.0; const SECT_NANG = 8; const SECT_NRAD = 2; const NSECT = SECT_NANG*SECT_NRAD

    # shape primitives (ported from PCATSim generate_qrm_thorax.jl; mm; mutate mask)
    stad!(m,xs,ys,l,cx,cy,hs,r)=(r2=r*r;@inbounds for j in eachindex(ys);dy=ys[j]-cy;for i in eachindex(xs);dx=abs(xs[i]-cx)-hs;dxc=dx>0 ? dx : 0.0;dxc*dxc+dy*dy≤r2&&(m[i,j]=l);end;end)
    circ!(m,xs,ys,l,cx,cy,r)=(r2=r*r;@inbounds for j in eachindex(ys);dy=ys[j]-cy;for i in eachindex(xs);dx=xs[i]-cx;dx*dx+dy*dy≤r2&&(m[i,j]=l);end;end)
    rectxy!(m,xs,ys,l,xa,xb,ya,yb)=(@inbounds for j in eachindex(ys);y=ys[j];(y<ya||y>yb)&&continue;for i in eachindex(xs);xa≤xs[i]≤xb&&(m[i,j]=l);end;end)
    function ellr!(m,xs,ys,l,cx,cy,a,b,θ);ct=cos(θ);st=sin(θ);a2=a*a;b2=b*b;@inbounds for j in eachindex(ys);dy=ys[j]-cy;for i in eachindex(xs);dx=xs[i]-cx;xr=dx*ct+dy*st;yr=-dx*st+dy*ct;(xr*xr)/a2+(yr*yr)/b2≤1&&(m[i,j]=l);end;end;end
    function fbl!(m,xs,ys,l,cx,ly,hcy,hr,ra,yt,yb);acy=ly+ra;dy=acy-hcy;L=sqrt((hr+ra)^2-dy^2);ar2=ra^2;@inbounds for j in eachindex(ys);y=ys[j];(y<yt||y>yb)&&continue;dyc=y-acy;d2=dyc*dyc;for i in eachindex(xs);x=xs[i];abs(x-cx)>L&&continue;dxr=x-(cx+L);dxl=x-(cx-L);(dxr*dxr+d2≥ar2&&dxl*dxl+d2≥ar2)&&(m[i,j]=l);end;end;end
    function fbc!(m,xs,ys,l,cx,cyt,rt,cyb,rb,ra,yt,yb);Δr=rt-rb;Σ=rt+rb+2ra;Δy=cyb-cyt;cya=(cyt+cyb)/2+Δr*Σ/(2Δy);dy=cya-cyt;L=sqrt((rt+ra)^2-dy^2);ar2=ra^2;@inbounds for j in eachindex(ys);y=ys[j];(y<yt||y>yb)&&continue;dyc=y-cya;d2=dyc*dyc;for i in eachindex(xs);x=xs[i];abs(x-cx)>L&&continue;dxr=x-(cx+L);dxl=x-(cx-L);(dxr*dxr+d2≥ar2&&dxl*dxl+d2≥ar2)&&(m[i,j]=l);end;end;end
    function tcol!(m,xs,ys,l,cx,hwt,hwb,yt,yb);ih=1.0/(yb-yt);@inbounds for j in eachindex(ys);y=ys[j];(y<yt||y>yb)&&continue;hw=hwt+(hwb-hwt)*(y-yt)*ih;for i in eachindex(xs);abs(xs[i]-cx)≤hw&&(m[i,j]=l);end;end;end
    scl!(m,xs,ys,l,cx,cy,r)=(r2=r*r;@inbounds for j in eachindex(ys);y=ys[j];y<cy&&continue;dy=y-cy;for i in eachindex(xs);dx=xs[i]-cx;dx*dx+dy*dy≤r2&&(m[i,j]=l);end;end)
    function varch!(m,xs,ys,l;apex_y,base_y,base_hw,sagitta=1.0,only_over=0xFF);sl=base_hw/(base_y-apex_y);cc=sagitta>0;ar=cc ? (base_hw^2+sagitta^2)/(2sagitta) : 0.0;acy=cc ? base_y+ar-sagitta : 0.0;ar2=ar*ar;@inbounds for j in eachindex(ys);y=ys[j];(y<apex_y||y>base_y)&&continue;hw=(y-apex_y)*sl;for i in eachindex(xs);x=xs[i];abs(x-185.0)>hw&&continue;if cc;dx=x-185.0;dy=y-acy;dx*dx+dy*dy<ar2&&continue;end;(only_over==0xFF||m[i,j]==only_over)&&(m[i,j]=l);end;end;end

    function _phantom(lbl,mats; nz=40)
        m3=repeat(reshape(lbl,size(lbl)...,1),1,1,nz); vc=VOXMM/10
        pc=BS.create_phantom_from_mask(Array{Int,3}(m3),mats,(vc,vc,vc))
        (cpu=pc,gpu=BS.Phantom(to_gpu(pc.mask),pc.materials,pc.voxel_size,pc.origin,pc.extent))
    end
    function build_thorax(comps; centres=HEARTC, ins_r=INS_R, radii=nothing, sectors=nothing, contrast=nothing, nz=40)
        vs=VOXMM; W=round(Int,370/vs); H=round(Int,270/vs)
        xs=[(i-0.5)*vs for i in 1:W]; ys=[(j-0.5)*vs for j in 1:H]; lbl=zeros(UInt8,W,H)
        stad!(lbl,xs,ys,0x05,185.0,135.0,50.0,125.0); stad!(lbl,xs,ys,0x02,185.0,135.0,50.0,100.0); stad!(lbl,xs,ys,0x01,185.0,135.0,50.0,80.0)
        fbl!(lbl,xs,ys,0x02,185.0,55.0,115.0,55.0,5.0,55.0,70.0); fbc!(lbl,xs,ys,0x02,185.0,115.0,55.0,190.0,20.0,5.0,167.55,175.86)
        circ!(lbl,xs,ys,0x02,HC_X,HC_Y,HEART_R_MM)
        if sectors===nothing
            @assert length(comps)==length(centres)
            for (k,(cx,cy)) in enumerate(centres); circ!(lbl,xs,ys,UInt8(ROD0-1+k),HC_X+cx,HC_Y+cy,radii===nothing ? ins_r : radii[k]); end
        else
            nang,nrad=sectors; @assert length(comps)==nang*nrad
            @inbounds for j in 1:H, i in 1:W
                x=xs[i]-HC_X; y=ys[j]-HC_Y; d2=x*x+y*y; d2≤SECT_R_MM^2 || continue
                ri=min(nrad-1,floor(Int,sqrt(d2)/(SECT_R_MM/nrad))); ai=min(nang-1,floor(Int,mod(atan(y,x),2π)/(2π/nang)))
                lbl[i,j]=UInt8(ROD0+ri*nang+ai)
            end
        end
        ribs=((185.0,45.0,0.0),(71.4,71.4,-π/4),(45.0,135.0,π/2),(71.4,198.6,π/4),(298.6,71.4,π/4),(325.0,135.0,π/2),(298.6,198.6,-π/4))
        for (a,b,θ) in ribs; ellr!(lbl,xs,ys,0x03,a,b,8.0,2.5,θ); end
        for (a,b,θ) in ribs; ellr!(lbl,xs,ys,0x04,a,b,7.0,1.5,θ); end
        circ!(lbl,xs,ys,0x03,185.0,190.0,20.0)
        varch!(lbl,xs,ys,0x02;apex_y=201.05,base_y=216.0,base_hw=42.0,sagitta=1.0,only_over=0x01)
        varch!(lbl,xs,ys,0x03;apex_y=201.05,base_y=212.67,base_hw=32.66,sagitta=1.0)
        circ!(lbl,xs,ys,0x03,172.39,205.54,1.5); circ!(lbl,xs,ys,0x03,197.61,205.54,1.5)
        tcol!(lbl,xs,ys,0x03,185.0,9.0,2.5,211.67,214.67); rectxy!(lbl,xs,ys,0x03,182.5,187.5,214.67,230.0)
        scl!(lbl,xs,ys,0x03,185.0,230.0,2.5); circ!(lbl,xs,ys,0x04,185.0,190.0,18.0)
        if contrast!==nothing; (cx,cy,cr)=contrast; circ!(lbl,xs,ys,0x07,cx,cy,cr); end
        mats=Dict{Int,BS.XA.Material}(0=>BS.XA.Materials.air,1=>BS.XA.Materials.lung,2=>BS.XA.Materials.muscle,
            3=>BS.XA.Materials.corticalbone,4=>BS.XA.Materials.marrow_red,5=>BS.XA.Materials.adipose)
        contrast!==nothing && (mats[7]=BS.XA.Materials.gammex_472_i5_0)
        for k in 1:length(comps); mats[ROD0-1+k]=wlp_material(comps[k]...;name="ins$k"); end
        merge(_phantom(lbl,mats;nz=nz), (centres_mm=centres, ins_r=ins_r))
    end
    # wide detector (1300 cols ≈ 415 mm scan FOV ⊃ 350 mm fat ring) · bowtie=:none (no peripheral fade)
    const SCANNER=BS.Scanner(source_to_isocenter=625.6,source_to_detector=1100.0,detector_rows=256,detector_cols=1300,
        detector_row_size=0.625,detector_col_size=0.6,focal_spot_width=1.0,focal_spot_length=1.0,
        target_angle=10.0,flat_filter_material=:aluminum,flat_filter_thickness=2.5,bowtie_filter=:none,
        detector_material=:lumex,detector_depth=3.0,fill_factor_row=0.9,fill_factor_col=0.9,electronic_noise=0,detection_gain=10.0)
    function run_acq(pg; views=984,collimation=2.5,matrix=(RECON_N,RECON_N,3),fov_cm=RECON_FOV_MM/10,z_cm=0.1875,seed=1234,zmed=1)
        plow=BS.CTProtocol(kVp=80,mA=407*0.65,views=views,rotation_time=0.5,collimation_mm=collimation,additional_filters=[("Al",4.5)])
        phigh=BS.CTProtocol(kVp=140,mA=405*0.35,views=views,rotation_time=0.5,collimation_mm=collimation,additional_filters=[("Al",4.5)])
        so=BS.SimOptions(fidelity=:eict,use_noise=true,use_fill_factor=false,use_optical_crosstalk=false,use_scatter=false,projector=:dd_fast,seed=seed)
        ro=BS.ReconOptions(matrix_size=matrix,fov_cm=fov_cm,z_cm=z_cm)
        _sim(p)=begin ws=BS.create_eict_workspace(SCANNER,p,so,ro,pg);BS.simulate!(ws,pg,p,so);r=(sino=Array(ws.sinogram),geom=ws.geom);ws=nothing;GC.gc(true);r end
        slo=_sim(plow);shi=_sim(phigh); iod=BS.XA.Elements.Iodine;wat=BS.XA.Materials.water
        eL,ŵL=BS.resolve_source_spectrum_full(so,plow;scanner=SCANNER,geom=slo.geom,phantom=pg)
        eH,ŵH=BS.resolve_source_spectrum_full(so,phigh;scanner=SCANNER,geom=shi.geom,phantom=pg)
        μρ(m,e)=Float32[Float32(BS.compute_mass_μ_at_energy(m,Float64(E))) for E in e]
        basis=(ŵ_L=ŵL,p_L=μρ(iod,eL),q_L=μρ(wat,eL),ŵ_H=ŵH,p_H=μρ(iod,eH),q_H=μρ(wat,eH))
        slo_g=to_gpu(Float32.(slo.sino));shi_g=to_gpu(Float32.(shi.sino)); sy=similar(slo_g);fill!(sy,0f0);sc=similar(slo_g);fill!(sc,0f0)
        cws=BS.create_cong_workspace(slo_g,basis);BS.apply_cong!(cws,sy,sc,slo_g,shi_g;water_basis=(a=0f0,c=1f0))
        siod=Array(sy);swat=Array(sc);slo_g=shi_g=sy=sc=cws=nothing;GC.gc(true)
        _fbp(s)=begin g=to_gpu(Float32.(s));ws=BS.create_fdk_recon_workspace(g,slo.geom,matrix;filter=BS.SoftFilter());r=Array(BS.reconstruct!(ws,g,slo.geom));ws=g=nothing;GC.gc(true);Float32.(r) end
        viod=_fbp(siod);vwat=_fbp(swat)
        zmed>0&&(viod=BS.apply_median_z(viod;adjacent_slices=zmed);vwat=BS.apply_median_z(vwat;adjacent_slices=zmed))
        ciod=viod.*1000f0
        (hu40=BS.synth_vmi_2basis(vwat,ciod;energy_keV=E40),hu70=BS.synth_vmi_2basis(vwat,ciod;energy_keV=E70),geom=slo.geom)
    end
    function roi_cores(pc,geom,matrix,labels; radius_px=12)
        m3=BS.resample_to_recon(pc,geom,matrix;method=:nearest); midz=size(m3,3)÷2+1; m2=m3[:,:,midz]
        nx,ny=size(m2); out=Dict{Int,Vector{CartesianIndex{2}}}()
        for lab in labels; idx=findall(==(UInt8(lab)),m2); isempty(idx)&&continue
            cx=mean(getindex.(idx,1));cy=mean(getindex.(idx,2))
            out[lab]=[CartesianIndex(i,j) for j in 1:ny,i in 1:nx if (i-cx)^2+(j-cy)^2≤radius_px^2]; end
        (cores=out,midz=midz,m2=m2)
    end
    function collect_rois(acq,pc,label_comp; radius_px=12)
        rc=roi_cores(pc,acq.geom,(RECON_N,RECON_N,3),collect(keys(label_comp)); radius_px=radius_px); out=NamedTuple[]
        for (lab,c) in label_comp
            (haskey(rc.cores,lab)&&!isempty(rc.cores[lab]))||continue; ci=rc.cores[lab]
            v40=[Float64(acq.hu40[i,rc.midz]) for i in ci]; v70=[Float64(acq.hu70[i,rc.midz]) for i in ci]
            push!(out,(lab=lab,fw=c[1],fl=c[2],fp=c[3],v40=v40,v70=v70,m40=mean(v40),m70=mean(v70),s40=std(v40),s70=std(v70)))
        end
        (rois=out,midz=rc.midz,m2=rc.m2)
    end
    md"`build_thorax` (packed / sector / centred) · `run_acq` (1300-col, bowtie-free, 984-view) · `roi_cores`/`collect_rois`"
end

# ╔═╡ aaaa0011-0000-4000-8000-000000000011
md"## 5 · Inverse: calibration surface · noise · edge-preserving TV"

# ╔═╡ aaaa0012-0000-4000-8000-000000000012
begin
    poly2(h4,h7)=[1.0,h4,h7,h4^2,h7^2,h4*h7]; surf(c,h4,h7)=dot(c,poly2(h4,h7))
    quad_sigma(c,H)=c[1]*H^2+c[2]*H+c[3]
    fit_sigma_quad(hu,sig)=(X=hcat(hu.^2,hu,ones(length(hu)));c=X\sig;c[1]<0&&(Xa=hcat(hu,ones(length(hu)));ca=Xa\sig;c=[0.0,ca[1],ca[2]]);c)
    function metrics(t,r)
        mt,mr=mean(t),mean(r);st2=mean((t.-mt).^2);sr2=mean((r.-mr).^2);str=mean((t.-mt).*(r.-mr))
        (ccc=2str/(st2+sr2+(mt-mr)^2),slope=str/st2,int=mr-str/st2*mt,rmse=sqrt(mean((r.-t).^2)),r2=1-sum((r.-t).^2)/sum((t.-mt).^2))
    end
    # coupled edge-preserving Huber-TV on (f_l,f_p); w = optional σ_f data weight (1/σ_f²). Never Gaussian.
    function tv_coupled(yl,yp,mask; lambda=0.05,iters=25,eps=0.04,w=nothing)
        nx,ny=size(yl)
        fl=[mask[i,j] ? Float64(yl[i,j]) : 0.0 for i in 1:nx,j in 1:ny]; fp=[mask[i,j] ? Float64(yp[i,j]) : 0.0 for i in 1:nx,j in 1:ny]
        fl2=copy(fl);fp2=copy(fp); inb(i,j)=1≤i≤nx&&1≤j≤ny&&mask[i,j]
        smp(a,b)=(a=max(a,0.0);b=max(b,0.0);s=a+b;s>1 ? (a/s,b/s) : (a,b))
        for _ in 1:iters
            @inbounds for j in 1:ny,i in 1:nx
                mask[i,j] || (fl2[i,j]=fl[i,j];fp2[i,j]=fp[i,j];continue)
                wij = (w===nothing || !isfinite(w[i,j])) ? 1.0 : w[i,j]
                rl=wij*Float64(yl[i,j]);rp=wij*Float64(yp[i,j]);den=wij
                for (di,dj) in ((1,0),(-1,0),(0,1),(0,-1)); inb(i+di,j+dj)||continue
                    dl=fl[i+di,j+dj]-fl[i,j];dp=fp[i+di,j+dj]-fp[i,j];c=lambda/max(sqrt(dl^2+dp^2),eps)
                    rl+=c*fl[i+di,j+dj];rp+=c*fp[i+di,j+dj];den+=c; end
                fl2[i,j],fp2[i,j]=smp(rl/den,rp/den)
            end
            fl,fl2=fl2,fl;fp,fp2=fp2,fp
        end
        ([mask[i,j] ? fl[i,j] : NaN for i in 1:nx,j in 1:ny],[mask[i,j] ? fp[i,j] : NaN for i in 1:nx,j in 1:ny])
    end
    dpoly4(h4,h7)=[0.0,1.0,0.0,2h4,0.0,h7]; dpoly7(h4,h7)=[0.0,0.0,1.0,0.0,2h7,h4]
    label_centroid(m2,lab)=(idx=findall(==(UInt8(lab)),m2); (mean(getindex.(idx,1)),mean(getindex.(idx,2))))
    struct BayesPrior; μ_w::Float64; s_w::Float64; μ_l::Float64; s_l::Float64; α_p::Float64; θ_p::Float64; end
    bayes_prior_broad(comps;s_wl=0.15,fp_shape=1.2,fp_scale=0.10)=(fw=[c[1] for c in comps];fl=[c[2] for c in comps];BayesPrior(mean(fw),s_wl,mean(fl),s_wl,fp_shape,fp_scale))
    md"`surf` (calibration) · `fit_sigma_quad` · `metrics` (CCC…) · `tv_coupled` (σ_f Huber-TV) · `bayes_prior_broad`"
end

# ╔═╡ aaaa0013-0000-4000-8000-000000000013
md"""## 6 · Run sims + calibrate + decode (cached)
4 calibration + 1 map thorax; **5 circular-test + 4 sector-test** held-out (n = 65 + 64 = 129 ROIs); 4 centred
integrated-HU sims. First run ≈ 20 min on GPU; results cache to `wlp_*_cache_70_150.jls`. Delete those to re-sim."""

# ╔═╡ aaaa0014-0000-4000-8000-000000000014
begin
    const PTAG   = "$(Int(E40))_$(Int(E70))"                       # pair-derived cache tag (each keV pair its own cache)
    const CACHE  = joinpath(@__DIR__, "wlp_sim_cache_$(PTAG).jls") # calibration + delivered-map thorax
    const TCACHE = joinpath(@__DIR__, "wlp_test_cache_$(PTAG).jls")# circular held-out test (5 sims)
    const ICACHE = joinpath(@__DIR__, "wlp_int_cache_$(PTAG).jls")
    const SCACHE = joinpath(@__DIR__, "wlp_sect_cache_$(PTAG).jls")
    const CORE_RPX = round(Int, INS_R/RECON_PX_MM - EROSION_PX)   # eroded interior core ≈ 225 mm²
    const IFL = 0.85
    # one packed-thorax acquisition → NHEART eroded insert cores (shared by calibration + circular test)
    _cores(seed, cseed) = begin
        comps = diverse_comps(cseed, NHEART); ph = build_thorax(comps); acq = run_acq(ph.gpu; seed=seed)
        lc = Dict(ROD0-1+k => comps[k] for k in 1:NHEART); r = collect_rois(acq, ph.cpu, lc; radius_px=CORE_RPX).rois
        img = (; seed, hu40=Array(acq.hu40), hu70=Array(acq.hu70), comps)   # full z-stack CT for raw export (shared geometry ⇒ map_m2 labels)
        ph=nothing; GC.gc(true); (rois=r, img=img)
    end
    # ── calibration (4 sims) + delivered-map thorax ──
    if !isfile(CACHE)
        println("── running calibration + map thorax ──"); Random.seed!(1)
        calrois = NamedTuple[]; calsims = NamedTuple[]
        for (si,seed) in enumerate((11,12,13,14)); c=_cores(seed, 1000+si); append!(calrois, c.rois); push!(calsims, c.img); end
        mcomps = diverse_comps(777, NHEART); mph = build_thorax(mcomps); macq = run_acq(mph.gpu; seed=999)
        mrc = roi_cores(mph.cpu, macq.geom, (RECON_N,RECON_N,3), collect(ROD0:(ROD0+NHEART-1)))
        map40v = Array(macq.hu40); map70v = Array(macq.hu70); map_m2 = mrc.m2; mph=nothing; GC.gc(true)   # full z-stack; 2D map derived below
        serialize(CACHE, (; calrois, map40v, map70v, map_m2, mcomps, calsims))
    end
    Dm = deserialize(CACHE); calrois=Dm.calrois; map40v,map70v,map_m2,mcomps = Dm.map40v,Dm.map70v,Dm.map_m2,Dm.mcomps; calsims=Dm.calsims
    mmid = size(map70v,3)÷2+1; map40 = map40v[:,:,mmid]; map70 = map70v[:,:,mmid]   # notebook figures/decode use the 2D mid slice
    # ── circular held-out test (5 sims → 65 ROIs; own cache) ──
    if !isfile(TCACHE)
        println("── running circular test thorax (5 sims) ──")
        testrois = NamedTuple[]; testsims = NamedTuple[]
        for (si,seed) in enumerate((201,202,203,204,205)); c=_cores(seed, 91260708+si); append!(testrois, c.rois); push!(testsims, c.img); end
        serialize(TCACHE, (; testrois, testsims))
    end
    Dt = deserialize(TCACHE); testrois = Dt.testrois; testsims = Dt.testsims
    # ── integrated-HU size series (one centred fat insert per sim) ──
    if !isfile(ICACHE)
        println("── running integrated-HU size series ──")
        isims = NamedTuple[]
        for r in SPARSE_RADII
            ip = build_thorax([(1-IFL-0.05,IFL,0.05)]; centres=[(0.0,0.0)], radii=[r]); ia = run_acq(ip.gpu; seed=Int(round(500+r)))
            im = size(ia.hu70,3)÷2+1; m2 = roi_cores(ip.cpu, ia.geom, (RECON_N,RECON_N,3), [ROD0]).m2
            push!(isims, (r=r, hu40=Array(ia.hu40[:,:,im]), hu70=Array(ia.hu70[:,:,im]), m2=m2)); ip=nothing; GC.gc(true)
        end
        serialize(ICACHE, (; isims))
    end
    isims = deserialize(ICACHE).isims
    # ── sector validation thorax (held-out shape) ──
    if !isfile(SCACHE)
        println("── running sector validation thorax ──")
        sectrois = NamedTuple[]; sectsims = NamedTuple[]
        for (si,seed) in enumerate((401,402,405,406))                # 4 sims → 64 ROIs
            sc = diverse_comps(70260708+si, NSECT); sp = build_thorax(sc; sectors=(SECT_NANG,SECT_NRAD)); sa = run_acq(sp.gpu; seed=seed)
            lc = Dict(ROD0-1+k => sc[k] for k in 1:NSECT); append!(sectrois, collect_rois(sa, sp.cpu, lc; radius_px=7).rois)
            push!(sectsims, (; seed, hu40=Array(sa.hu40), hu70=Array(sa.hu70), comps=sc)); sp=nothing; GC.gc(true)   # full z-stack
        end
        scomps = diverse_comps(70260800, NSECT); smph = build_thorax(scomps; sectors=(SECT_NANG,SECT_NRAD)); smacq = run_acq(smph.gpu; seed=403)
        smrc = roi_cores(smph.cpu, smacq.geom, (RECON_N,RECON_N,3), collect(ROD0:(ROD0+NSECT-1)))
        smap40v = Array(smacq.hu40); smap70v = Array(smacq.hu70); smap_m2 = smrc.m2; smph=nothing; GC.gc(true)   # full z-stack; 2D map derived below
        serialize(SCACHE, (; sectrois, smap40v, smap70v, smap_m2, scomps, sectsims))
    end
    DS = deserialize(SCACHE); sectrois=DS.sectrois; smap40v,smap70v,smap_m2,scomps = DS.smap40v,DS.smap70v,DS.smap_m2,DS.scomps; sectsims=DS.sectsims
    smid = size(smap70v,3)÷2+1; smap40 = smap40v[:,:,smid]; smap70 = smap70v[:,:,smid]   # notebook figures/decode use the 2D mid slice

    # ── calibration: quadratic surface (point accuracy) + AFFINE lipid (integrals) + noise ladder ──
    m40c=[r.m40 for r in calrois]; m70c=[r.m70 for r in calrois]
    fwc=[r.fw for r in calrois]; flc=[r.fl for r in calrois]; fpc=[r.fp for r in calrois]
    Xc=reduce(vcat,[poly2(m40c[i],m70c[i])' for i in eachindex(m40c)]); cw=Xc\fwc; cl=Xc\flc; cp=Xc\fpc
    Xa=hcat(ones(length(m40c)),m40c,m70c); cl_aff=Xa\flc                 # affine f_l — linear ⇒ commutes with the PSF
    r2fit(c,y)=1-sum((surf.(Ref(c),m40c,m70c).-y).^2)/sum((y.-mean(y)).^2)
    sc40=fit_sigma_quad(m40c,[r.s40 for r in calrois]); sc70=fit_sigma_quad(m70c,[r.s70 for r in calrois])
    res40=vcat([r.v40.-r.m40 for r in calrois]...); res70=vcat([r.v70.-r.m70 for r in calrois]...); ρ=cor(res40,res70)
    adipose=draw_wlp(20260708,2000); prior=bayes_prior_broad(adipose)
    decode(a,b)=(x=surf(cw,a,b);y=surf(cl,a,b);z=surf(cp,a,b);s=x+y+z;(x/s,y/s,z/s))
    aff_l(a,b)=cl_aff[1]+cl_aff[2]*a+cl_aff[3]*b

    # ── point accuracy: combined circular + sector, per-voxel decode over the eroded core (GT only locates) ──
    allrois=vcat(testrois,sectrois); geomtag=vcat(fill(:circular,length(testrois)),fill(:sector,length(sectrois)))
    tfw=[r.fw for r in allrois];tfl=[r.fl for r in allrois];tfp=[r.fp for r in allrois]
    pvox(r)=[decode(r.v40[j],r.v70[j]) for j in eachindex(r.v40)]
    pfw=[mean(getindex.(pvox(r),1)) for r in allrois]; pfl=[mean(getindex.(pvox(r),2)) for r in allrois]; pfp=[mean(getindex.(pvox(r),3)) for r in allrois]
    semfw=[std(getindex.(pvox(r),1))/sqrt(length(r.v40)) for r in allrois]
    semfl=[std(getindex.(pvox(r),2))/sqrt(length(r.v40)) for r in allrois]; semfp=[std(getindex.(pvox(r),3))/sqrt(length(r.v40)) for r in allrois]
    mw=metrics(tfw,pfw);ml=metrics(tfl,pfl);mp=metrics(tfp,pfp)
    pfl_pool=[decode(r.m40,r.m70)[2] for r in allrois]; ml_pool=metrics(tfl,pfl_pool)
    dHU70=[abs(mix_hu(pfw[i],pfl[i],pfp[i],E70)-mix_hu(tfw[i],tfl[i],tfp[i],E70)) for i in eachindex(tfw)]

    # ── delivered map: per-voxel decode over gated soft tissue + σ_f-weighted edge-preserving Huber-TV ──
    const SOFT_HU_LO, SOFT_HU_HI = -300.0, 250.0
    function sigma_f_weight(h40,h70)
        w=fill(NaN,size(h70))
        for I in CartesianIndices(h70); (SOFT_HU_LO<h70[I]<SOFT_HU_HI)||continue
            h4=Float64(h40[I]);h7=Float64(h70[I]); s4=quad_sigma(sc40,h4);s7=quad_sigma(sc70,h7)
            g4l=dot(cl,dpoly4(h4,h7));g7l=dot(cl,dpoly7(h4,h7)); g4p=dot(cp,dpoly4(h4,h7));g7p=dot(cp,dpoly7(h4,h7))
            vl=g4l^2*s4^2+g7l^2*s7^2+2ρ*g4l*g7l*s4*s7; vp=g4p^2*s4^2+g7p^2*s7^2+2ρ*g4p*g7p*s4*s7
            w[I]=1.0/max(vl+vp,1e-6)
        end; w
    end
    function fullfield(h40,h70)
        fw=fill(NaN,size(h70));fl=copy(fw);fp=copy(fw)
        for I in CartesianIndices(h70); (SOFT_HU_LO<h70[I]<SOFT_HU_HI)||continue
            d=decode(Float64(h40[I]),Float64(h70[I])); fw[I]=d[1];fl[I]=d[2];fp[I]=d[3]; end
        (fw,fl,fp)
    end
    function deliver(m40,m70,m2,comps)
        f0=fullfield(m40,m70); gate=.!isnan.(f0[2]); w=sigma_f_weight(m40,m70)
        fl_tv,fp_tv=tv_coupled(f0[2],f0[3],gate; lambda=0.05,iters=25,eps=0.04,w=w)
        fw_tv=map((a,b)-> isnan(a) ? NaN : 1-a-b, fl_tv, fp_tv)
        rec=cat(fw_tv,fl_tv,fp_tv;dims=3); tru=fill(NaN,size(m2)...,3); recgt=fill(NaN,size(m2)...,3)
        for k in 1:length(comps); lab=ROD0-1+k
            idx=findall(==(UInt8(lab)),m2); isempty(idx)&&continue
            d=decode(mean(Float64(m40[I]) for I in idx),mean(Float64(m70[I]) for I in idx))
            for I in idx; tru[I,1],tru[I,2],tru[I,3]=comps[k]; recgt[I,1],recgt[I,2],recgt[I,3]=d; end
        end
        (rec=rec,tru=tru,recgt=recgt)
    end
    circ = deliver(map40,map70,map_m2,mcomps)
    sect = deliver(smap40,smap70,smap_m2,scomps)
    recmap=circ.rec; truemap=circ.tru; recmap_gt=circ.recgt

    # ── integrated-HU: EXCESS lipid over local muscle; conservation recovers ∫(f_l−bg) without the boundary ──
    const FIXED_MARGIN_PX = 8.0
    IFAT=(1-IFL-0.05,IFL,0.05); FAT_AFF=aff_l(mix_hu(IFAT...,E40),mix_hu(IFAT...,E70))   # affine content; vs IFL = decode bias
    integ = NamedTuple[]
    for s in isims
        i40=s.hu40; i70=s.hu70; m2=s.m2; r_mm=s.r; rpx=r_mm/RECON_PX_MM; (cx,cy)=label_centroid(m2,ROD0)
        qb(i,j)=[1.0,i-cx,j-cy,(i-cx)^2,(j-cy)^2,(i-cx)*(j-cy)]                          # global quadratic muscle bg (cupping)
        hm=[CartesianIndex(i,j) for i in axes(i40,1),j in axes(i40,2) if m2[i,j]==0x02 && (i-cx)^2+(j-cy)^2≤(60.0/RECON_PX_MM)^2]
        cg=reduce(vcat,[qb(I[1],I[2])' for I in hm])\[aff_l(Float64(i40[I]),Float64(i70[I])) for I in hm]
        bgf(i,j)=dot(cg,qb(i,j)); bg0=bgf(cx,cy); truelip=π*r_mm^2*(FAT_AFF-bg0)
        margins=0.0:1.0:16.0; recov=Float64[]
        for mg in margins
            R=rpx+mg; acc=0.0
            for i in axes(i40,1), j in axes(i40,2)
                (i-cx)^2+(j-cy)^2≤R^2 || continue; (SOFT_HU_LO<i70[i,j]<SOFT_HU_HI) || continue
                acc += aff_l(Float64(i40[i,j]),Float64(i70[i,j])) - bgf(i,j)
            end
            push!(recov, acc*RECON_PX_MM^2)
        end
        bi=findfirst(==(FIXED_MARGIN_PX),margins)
        push!(integ,(r=r_mm,fl=IFL,truelip=truelip,bg=bg0,naivelip=recov[1],intlip=recov[bi],margins=collect(margins),recov=recov))
    end
    @printf("PAIR %g/%g keV | cal n=%d R²(f_w)=%.3f ρ=%.3f | TEST n=%d: f_w CCC=%.3f f_l CCC=%.3f f_p CCC=%.3f | cond(G)=%.1f | integrated %.0f–%.0f%% vs naive %.0f–%.0f%%\n",
        E40,E70,length(calrois),r2fit(cw,fwc),ρ,length(allrois),mw.ccc,ml.ccc,mp.ccc,
        cond([PL[1]-PW[1] PP[1]-PW[1]; PL[2]-PW[2] PP[2]-PW[2]]),
        100*minimum(r.intlip/r.truelip for r in integ),100*maximum(r.intlip/r.truelip for r in integ),
        100*minimum(r.naivelip/r.truelip for r in integ),100*maximum(r.naivelip/r.truelip for r in integ))
    Markdown.parse("cal n=$(length(calrois)), R²(f_w)=$(round(r2fit(cw,fwc),digits=3)); **TEST n=$(length(allrois))** ($(count(==(:circular),geomtag)) circular + $(count(==(:sector),geomtag)) sector) — f_w CCC=**$(round(mw.ccc,digits=3))**, f_l CCC=**$(round(ml.ccc,digits=3))**, f_p CCC=**$(round(mp.ccc,digits=3))**; ρ=$(round(ρ,digits=2)); integrated-HU recovers $(round(Int,100*minimum(r.intlip/r.truelip for r in integ)))–$(round(Int,100*maximum(r.intlip/r.truelip for r in integ)))% vs naive $(round(Int,100*minimum(r.naivelip/r.truelip for r in integ)))–$(round(Int,100*maximum(r.naivelip/r.truelip for r in integ)))%.")
end

# ╔═╡ aaaa0015-0000-4000-8000-000000000015
md"## 7 · Figures"

# ╔═╡ aaaa0016-0000-4000-8000-000000000016
let f=CM.Figure(size=(600,560)), flcol=[r.fl for r in calrois]
    ax=CM.Axis(f[1,1];xlabel="HU$(Int(E40))",ylabel="HU$(Int(E70))",title="Barycentric triangle · $(Int(E40)) vs $(Int(E70)) keV",aspect=CM.DataAspect())
    CM.poly!(ax,[CM.Point2f(PW...),CM.Point2f(PL...),CM.Point2f(PP...)];color=(:steelblue,0.15),strokecolor=:gray,strokewidth=1)
    sc=CM.scatter!(ax,m40c,m70c;color=flcol,colormap=:viridis,markersize=9)
    for (p,t) in ((PW,"W"),(PL,"L"),(PP,"P")); CM.scatter!(ax,[p[1]],[p[2]];marker=:diamond,color=:black,markersize=13); CM.text!(ax,p[1],p[2];text=t,fontsize=16,align=(:center,:bottom)); end
    CM.Colorbar(f[1,2],sc;label="true f_l")
    CM.Label(f[0,:],"Calibration cores fall inside the theoretical W/L/P triangle";fontsize=13,font=:bold)
    safe_save(joinpath(ASSET,"fig1_triangle.png"),f); f
end

# ╔═╡ aaaa0017-0000-4000-8000-000000000017
let f=CM.Figure(size=(1050,460))
    ax=CM.Axis(f[1,1];xlabel="HU",ylabel="σ (HU)",title="σ(HU) per energy — convex-quadratic")
    for (hu,sg,cc,e,col) in ((m40c,[r.s40 for r in calrois],sc40,E40,:tomato),(m70c,[r.s70 for r in calrois],sc70,E70,:royalblue))
        CM.scatter!(ax,hu,sg;color=col,markersize=8,label="$(Int(e)) keV"); g=range(minimum(hu),maximum(hu),100); CM.lines!(ax,g,quad_sigma.(Ref(cc),g);color=col)
    end
    CM.axislegend(ax;position=:rt)
    ax2=CM.Axis(f[1,2];xlabel="resid HU$(Int(E40))",ylabel="resid HU$(Int(E70))",title="inter-energy ρ=$(round(ρ,digits=2))")
    idx=rand(1:length(res40),min(3000,length(res40))); CM.scatter!(ax2,res40[idx],res70[idx];markersize=3,color=(:purple,0.3))
    safe_save(joinpath(ASSET,"fig2_noise.png"),f); f
end

# ╔═╡ aaaa0018-0000-4000-8000-000000000018
let f=CM.Figure(size=(1520,430))
    ax=CM.Axis(f[1,1];title="VMI $(Int(E70)) keV",aspect=CM.DataAspect(),yreversed=true); CM.hidedecorations!(ax); CM.heatmap!(ax,map70;colormap=:grays,colorrange=(-200,300))
    for (col,(img,ttl)) in enumerate(((recmap[:,:,1],"f_w"),(recmap[:,:,2],"f_l"),(recmap[:,:,3],"f_p")))
        ax2=CM.Axis(f[1,col+1];title=ttl,aspect=CM.DataAspect(),yreversed=true); CM.hidedecorations!(ax2); CM.heatmap!(ax2,img;colormap=:jet,colorrange=(0,1))
    end
    CM.Colorbar(f[1,5];colormap=:jet,colorrange=(0,1),label="volume fraction")
    CM.Label(f[0,:],"Delivered map — per-voxel decode + σ_f-weighted Huber-TV, boundary-agnostic (lung & bone HU-gated out)";fontsize=13,font=:bold)
    safe_save(joinpath(ASSET,"fig3_delivered_map.png"),f); f
end

# ╔═╡ aaaa0019-0000-4000-8000-000000000019
let f=CM.Figure(size=(1250,440))
    hi=findall(!isnan,truemap[:,:,2]); ci=extrema(getindex.(hi,1)); cj=extrema(getindex.(hi,2)); pad=28
    rI=max(1,ci[1]-pad):min(size(map_m2,1),ci[2]+pad); rJ=max(1,cj[1]-pad):min(size(map_m2,2),cj[2]+pad)
    for (col,(img,ttl)) in enumerate(((truemap[rI,rJ,2],"true f_l (GT regions)"),(recmap[rI,rJ,2],"honest (per-voxel + TV)"),(recmap_gt[rI,rJ,2],"optimistic (GT-boundary pooled)")))
        ax=CM.Axis(f[1,col];title=ttl,aspect=CM.DataAspect(),yreversed=true); CM.hidedecorations!(ax); hm=CM.heatmap!(ax,img;colormap=:jet,colorrange=(0,1)); col==3&&CM.Colorbar(f[1,4],hm;label="f_l")
    end
    CM.Label(f[0,:],"f_l over the heart: honest per-voxel+TV keeps real texture & PVE edges; GT-pooled is flat (uses the boundary we don't have on real fat)";fontsize=12,font=:bold)
    safe_save(joinpath(ASSET,"fig4_honest_vs_pooled.png"),f); f
end

# ╔═╡ aaaa0020-0000-4000-8000-000000000020
let f=CM.Figure(size=(1300,460))
    for (col,(t,p,mt,sem,nm)) in enumerate(((tfw,pfw,mw,semfw,"f_w"),(tfl,pfl,ml,semfl,"f_l"),(tfp,pfp,mp,semfp,"f_p")))
        lo=min(minimum(t),minimum(p));hi=max(maximum(t),maximum(p))
        ax=CM.Axis(f[1,col];xlabel="true $nm",ylabel="recovered",title=nm,aspect=CM.DataAspect(),limits=(lo,hi,lo,hi))
        CM.lines!(ax,[lo,hi],[lo,hi];color=:gray,linestyle=:dash)
        CM.errorbars!(ax,t,p,sem;color=(:black,0.55),whiskerwidth=8,linewidth=1.2)   # capped error bars, all 3 materials
        CM.scatter!(ax,t,p;color=[g==:circular ? :steelblue : :orange for g in geomtag],markersize=8)
        CM.text!(ax,lo+0.03*(hi-lo),hi-0.05*(hi-lo);text=@sprintf("CCC %.3f\nslope %.2f\nRMSE %.3f\nR² %.3f",mt.ccc,mt.slope,mt.rmse,mt.r2),align=(:left,:top),fontsize=11)
    end
    CM.Label(f[0,:],"Recovered vs true (n=$(length(allrois)): $(count(==(:circular),geomtag)) circular + $(count(==(:sector),geomtag)) sector · per-voxel decode · eroded-core pool) — blue=circular, orange=sector; error bars = SE of the ROI mean";fontsize=12,font=:bold)
    safe_save(joinpath(ASSET,"fig5_scatter.png"),f); f
end

# ╔═╡ aaaa0021-0000-4000-8000-000000000021
let f=CM.Figure(size=(1250,470)), rr=[r.r for r in integ]
    ax=CM.Axis(f[1,1];xlabel="fat object radius (mm)",ylabel="recovered / true excess lipid",title="Object-extent measure vs integrated-HU")
    CM.scatterlines!(ax,rr,[r.naivelip/r.truelip for r in integ];color=:tomato,markersize=13,label="naive (object extent)")
    CM.scatterlines!(ax,rr,[r.intlip/r.truelip for r in integ];color=:seagreen,markersize=13,label="integrated (+$(Int(FIXED_MARGIN_PX)) px skirt)")
    CM.hlines!(ax,[1.0];color=:gray,linestyle=:dash); CM.axislegend(ax;position=:rb)
    ax2=CM.Axis(f[1,2];xlabel="integration margin (recon-px)",ylabel="recovered / true",title="Conservation vs margin")
    for r in integ; CM.lines!(ax2,r.margins,r.recov./r.truelip;linewidth=2,label=@sprintf("r=%.0f mm",r.r)); end
    CM.hlines!(ax2,[1.0];color=:gray,linestyle=:dash); CM.vlines!(ax2,[FIXED_MARGIN_PX];color=(:black,0.3),linestyle=:dot); CM.axislegend(ax2;position=:rb)
    CM.Label(f[0,:],"Partial volume makes the object-extent measure under-report small fat; integrated-HU recovers it via conservation (bg = local muscle)";fontsize=12,font=:bold)
    safe_save(joinpath(ASSET,"fig6_integrated_hu.png"),f); f
end

# ╔═╡ aaaa0022-0000-4000-8000-000000000022
let f=CM.Figure(size=(1520,430))
    ax=CM.Axis(f[1,1];title="VMI $(Int(E70)) keV (sector)",aspect=CM.DataAspect(),yreversed=true); CM.hidedecorations!(ax); CM.heatmap!(ax,smap70;colormap=:grays,colorrange=(-200,300))
    for (col,(img,ttl)) in enumerate(((sect.rec[:,:,1],"f_w"),(sect.rec[:,:,2],"f_l"),(sect.rec[:,:,3],"f_p")))
        ax2=CM.Axis(f[1,col+1];title=ttl,aspect=CM.DataAspect(),yreversed=true); CM.hidedecorations!(ax2); CM.heatmap!(ax2,img;colormap=:jet,colorrange=(0,1))
    end
    CM.Colorbar(f[1,5];colormap=:jet,colorrange=(0,1),label="volume fraction")
    CM.Label(f[0,:],"Delivered map — sector validation phantom (per-voxel decode + σ_f Huber-TV, boundary-agnostic)";fontsize=13,font=:bold)
    safe_save(joinpath(ASSET,"fig7_delivered_map_sector.png"),f); f
end

# ╔═╡ aaaa0023-0000-4000-8000-000000000023
# GT vs recovered (jet 0–1) vs signed error (blue–white–red diverging, own colorbar) for ALL THREE materials.
# Circular phantom.
let f=CM.Figure(size=(1320,1050)), Dl=circ
    hi=findall(!isnan,Dl.tru[:,:,2]); ci=extrema(getindex.(hi,1)); cj=extrema(getindex.(hi,2)); pad=25
    rI=max(1,ci[1]-pad):min(size(Dl.tru,1),ci[2]+pad); rJ=max(1,cj[1]-pad):min(size(Dl.tru,2),cj[2]+pad)
    for (row,mat) in enumerate(("f_w","f_l","f_p"))
        tl=Dl.tru[rI,rJ,row]; rf=Dl.rec[rI,rJ,row]
        rl=[isnan(tl[i,j]) ? NaN : rf[i,j] for i in axes(tl,1),j in axes(tl,2)]
        er=[isnan(tl[i,j]) ? NaN : rf[i,j]-tl[i,j] for i in axes(tl,1),j in axes(tl,2)]
        for (col,(img,ttl,cm,cr)) in enumerate(((tl,"true",:jet,(0,1)),(rl,"recovered",:jet,(0,1)),(er,"error",CM.Reverse(:RdBu),(-0.3,0.3))))
            ax=CM.Axis(f[row,col];title=(row==1 ? ttl : ""),ylabel=(col==1 ? mat : ""),aspect=CM.DataAspect(),yreversed=true)
            CM.hidedecorations!(ax;label=false); CM.heatmap!(ax,img;colormap=cm,colorrange=cr)
        end
    end
    CM.Colorbar(f[:,4];colormap=:jet,colorrange=(0,1),label="fraction (true / recovered)")
    CM.Colorbar(f[:,5];colormap=CM.Reverse(:RdBu),colorrange=(-0.3,0.3),label="error (recovered − true)")
    CM.Label(f[0,:],"Circular phantom — true & recovered (jet 0–1) vs error (blue–white–red), f_w / f_l / f_p";fontsize=13,font=:bold)
    safe_save(joinpath(ASSET,"fig8_gt_rec_error_circular.png"),f); f
end

# ╔═╡ aaaa0024-0000-4000-8000-000000000024
# Same triad for the SECTOR validation phantom (held-out shape).
let f=CM.Figure(size=(1320,1050)), Dl=sect
    hi=findall(!isnan,Dl.tru[:,:,2]); ci=extrema(getindex.(hi,1)); cj=extrema(getindex.(hi,2)); pad=25
    rI=max(1,ci[1]-pad):min(size(Dl.tru,1),ci[2]+pad); rJ=max(1,cj[1]-pad):min(size(Dl.tru,2),cj[2]+pad)
    for (row,mat) in enumerate(("f_w","f_l","f_p"))
        tl=Dl.tru[rI,rJ,row]; rf=Dl.rec[rI,rJ,row]
        rl=[isnan(tl[i,j]) ? NaN : rf[i,j] for i in axes(tl,1),j in axes(tl,2)]
        er=[isnan(tl[i,j]) ? NaN : rf[i,j]-tl[i,j] for i in axes(tl,1),j in axes(tl,2)]
        for (col,(img,ttl,cm,cr)) in enumerate(((tl,"true",:jet,(0,1)),(rl,"recovered",:jet,(0,1)),(er,"error",CM.Reverse(:RdBu),(-0.3,0.3))))
            ax=CM.Axis(f[row,col];title=(row==1 ? ttl : ""),ylabel=(col==1 ? mat : ""),aspect=CM.DataAspect(),yreversed=true)
            CM.hidedecorations!(ax;label=false); CM.heatmap!(ax,img;colormap=cm,colorrange=cr)
        end
    end
    CM.Colorbar(f[:,4];colormap=:jet,colorrange=(0,1),label="fraction (true / recovered)")
    CM.Colorbar(f[:,5];colormap=CM.Reverse(:RdBu),colorrange=(-0.3,0.3),label="error (recovered − true)")
    CM.Label(f[0,:],"Sector phantom — true & recovered (jet 0–1) vs error (blue–white–red), f_w / f_l / f_p";fontsize=13,font=:bold)
    safe_save(joinpath(ASSET,"fig9_gt_rec_error_sector.png"),f); f
end

# ╔═╡ aaaa0025-0000-4000-8000-000000000025
Markdown.parse("""
## 8 · Conclusion

| fraction | CCC | slope | RMSE |
|---|---|---|---|
| **f_water** | **$(round(mw.ccc,digits=3))** | $(round(mw.slope,digits=2)) | $(round(mw.rmse,digits=3)) |
| f_lipid | $(round(ml.ccc,digits=3)) | $(round(ml.slope,digits=2)) | $(round(ml.rmse,digits=3)) |
| f_protein | $(round(mp.ccc,digits=3)) | $(round(mp.slope,digits=2)) | $(round(mp.rmse,digits=3)) |

Held-out **circular + sector** (n=$(length(allrois))). Detectability: **$(round(Int,100mean(dHU70.<5)))% of ROIs < 5 HU at $(Int(E70)) keV** (mean $(round(mean(dHU70),digits=1)) HU).

**keV pair — $(Int(E40))/$(Int(E70)).** This pair is intentionally ill-conditioned (150 keV is a clinically
standard VMI, but the two energies sit above the photoelectric-rich low-keV regime): the W/L/P triangle is a
near-collinear sliver, **cond(G) = $(round(cond([PL[1]-PW[1] PP[1]-PW[1]; PL[2]-PW[2] PP[2]-PW[2]]),digits=1))**
(≈3.5× a low-keV pair). Yet the held-out **ROI CCC is unaffected** — the √N eroded-core pooling absorbs the
per-voxel conditioning penalty, and the inter-energy noise correlation ρ=$(round(ρ,digits=2)) is low enough to
help the separation. The penalty surfaces only in the per-voxel maps and the integrated-HU total; conditioning
number alone overpredicts it. The pair is a single `WLP_PAIR` knob.

**Point accuracy** is excellent on the eroded interior cores (all CCC ≈ 0.99) and, as expected on a uniform
phantom, per-voxel vs pool-then-decode barely differ there. The honesty cost of the ground-truth boundary shows
up in the **delivered map** (fig 3–4): the boundary-agnostic per-voxel+TV map keeps real texture and PVE edges,
whereas the GT-pooled map is flat because it uses a boundary real fat doesn't provide.

**Integrated-HU** (fig 6) is the answer to partial-volume underestimation of small fat: the object-extent
measure loses $(round(Int,100-100*minimum(r.naivelip/r.truelip for r in integ)))% of a 4 mm fat object, while
integrating an affine decode over the object+skirt with a local muscle background recovers it to
$(round(Int,100*minimum(r.intlip/r.truelip for r in integ)))–$(round(Int,100*maximum(r.intlip/r.truelip for r in integ)))% via mass conservation. The affine decode carries a separate
$(round(Int,100*(FAT_AFF/IFL-1)))% composition bias at f_l=$(IFL) (linearization error), common to both estimators. Real-data caveats:
integrated-HU fixes the *numerator* (total lipid) — a region *mean* still needs a segmentation denominator; the
local background must be a field (not a constant) because PVAT muscle isn't uniform; and conservation is exact
only for linear/FBP recon, so a clinical DLIR/QIR transfer must re-earn it empirically.
""")

# ╔═╡ Cell order:
# ╟─aaaa0002-0000-4000-8000-000000000002
# ╟─aaaa0003-0000-4000-8000-000000000003
# ╠═aaaa0001-0000-4000-8000-000000000001
# ╠═aaaa0004-0000-4000-8000-000000000004
# ╟─aaaa0005-0000-4000-8000-000000000005
# ╠═aaaa0006-0000-4000-8000-000000000006
# ╟─aaaa0007-0000-4000-8000-000000000007
# ╠═aaaa0008-0000-4000-8000-000000000008
# ╟─aaaa0009-0000-4000-8000-000000000009
# ╠═aaaa0010-0000-4000-8000-000000000010
# ╟─aaaa0011-0000-4000-8000-000000000011
# ╠═aaaa0012-0000-4000-8000-000000000012
# ╟─aaaa0013-0000-4000-8000-000000000013
# ╠═aaaa0014-0000-4000-8000-000000000014
# ╟─aaaa0015-0000-4000-8000-000000000015
# ╠═aaaa0016-0000-4000-8000-000000000016
# ╠═aaaa0017-0000-4000-8000-000000000017
# ╠═aaaa0018-0000-4000-8000-000000000018
# ╠═aaaa0019-0000-4000-8000-000000000019
# ╠═aaaa0020-0000-4000-8000-000000000020
# ╠═aaaa0021-0000-4000-8000-000000000021
# ╠═aaaa0022-0000-4000-8000-000000000022
# ╠═aaaa0023-0000-4000-8000-000000000023
# ╠═aaaa0024-0000-4000-8000-000000000024
# ╟─aaaa0025-0000-4000-8000-000000000025
