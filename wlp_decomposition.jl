### A Pluto.jl notebook ###
# v0.1.0

using Markdown
using InteractiveUtils

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

# ╔═╡ aaaa0002-0000-4000-8000-000000000002
md"""
# Water–Lipid–Protein Material Decomposition on a QRM-Thorax — a pure-physics study

Recover the **volumetric fractions** ``(f_w, f_l, f_p)`` of water/lipid/protein mixtures from dual-energy CT
on a **stadium QRM-thorax phantom** (two lungs split by a mediastinal muscle column,
ribs, spine, and a heart cavity holding the material inserts). Everything is inline (the only data file is the
Woodard adipose CSV for the prior): mix materials by volume fraction, simulate 80/140-kVp DECT with
[BasisSimulator.jl](https://github.com/MolloiLab/BasisSimulator.jl) **(v0.8.0, `:dd_fast`)**, synthesize VMI at
**70 and 150 keV** (the `WLP_PAIR` knob; 150 keV is a clinically standard VMI), and invert.

**Two complementary products.**
1. A **quadratic calibration surface** ``f=\mathrm{poly}_2(\mathrm{HU}_{70},\mathrm{HU}_{150})`` for per-voxel
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
    const ELO, EHI = WLP_PAIR             # low / high energy of the pair
    const PL = (theo_hu(LIPID,ELO), theo_hu(LIPID,EHI)); const PP = (theo_hu(PROTEIN,ELO), theo_hu(PROTEIN,EHI)); const PW = (0.0, 0.0)
    # md-macro interpolation breaks on repeated `=$` adjacency — interpolate pre-formatted strings
    pl_str = string(round.(Int, PL)); pp_str = string(round.(Int, PP))
    md"Endpoints (theoretical HU): water = (0,0), lipid = $pl_str, protein = $pp_str."
end

# ╔═╡ aaaa0007-0000-4000-8000-000000000007
md"## 3 · Composition generation"

# ╔═╡ aaaa0008-0000-4000-8000-000000000008
begin
    const ADI = ("healthy","obese","reduced","comparison","unspecified")
    mass_to_vol(w,l,p) = (v=(w/ΡW,l/ΡL,p/ΡP); s=sum(v); v./s)
    # Adipose composition vs lipid fraction, digitized from Woodard & White (1986) Fig. 1
    # (Br J Radiol 59:1209-1219), which compiles Mitchell 1945, Forbes 1953/1956,
    # Entenman 1958, Pawan & Clode 1960, Thomas 1962, Morse & Soeldner 1963, Baker 1969.
    # Inlined so this notebook is standalone. Provenance: paper = table value, figure = digitized point.
    const ADIPOSE_CSV = """
    component,study,source,lipid_pct,component_pct,provenance,state
    water,1,Mitchell et al. 1945,42.44,50.09,paper,healthy
    water,9,Woodard 1986,52.9,25.7,figure,healthy
    water,8,Baker 1969,56.0,41.4,figure,healthy
    water,6,Thomas 1962,56.6,23.1,figure,comparison
    water,7,Morse & Soeldner 1963,56.6,28.8,figure,comparison
    water,4,Entenman et al. 1958,62.3,32.4,paper,reduced
    water,8,Baker 1969,64.9,30.6,figure,healthy
    water,8,Baker 1969,65.6,33.1,figure,healthy
    water,5,Pawan & Clode 1960,68.3,28.5,figure,comparison
    water,9,Woodard 1986,69.4,20.5,figure,healthy
    water,9,Woodard 1986,70.0,18.3,figure,healthy
    water,6,Thomas 1962,70.3,28.5,figure,comparison
    water,0,(unlabeled in FIG.1),71.4,26.0,figure,unspecified
    water,2,Forbes et al. 1953,71.57,23.02,paper,healthy
    water,0,(unlabeled in FIG.1),72.4,26.0,figure,unspecified
    water,0,(unlabeled in FIG.1),73.2,25.1,figure,unspecified
    water,7,Morse & Soeldner 1963,77.2,20.0,figure,comparison
    water,7,Morse & Soeldner 1963,78.1,19.6,figure,comparison
    water,8,Baker 1969,78.3,17.9,figure,healthy
    water,3,Forbes et al. 1956,78.35,16.76,paper,healthy
    water,4,Entenman et al. 1958,78.9,17.7,paper,reduced
    water,4,Entenman et al. 1958,79.2,18.0,paper,obese
    water,8,Baker 1969,79.4,18.1,figure,healthy
    water,9,Woodard 1986,80.0,10.7,figure,healthy
    water,9,Woodard 1986,80.05,9.25,figure,healthy
    water,9,Woodard 1986,80.25,8.75,figure,healthy
    water,4,Entenman et al. 1958,85.7,12.5,paper,obese
    water,8,Baker 1969,86.7,11.2,figure,healthy
    water,8,Baker 1969,87.6,10.3,figure,healthy
    water,8,Baker 1969,87.6,12.0,figure,healthy
    water,7,Morse & Soeldner 1963,88.6,9.5,figure,comparison
    protein,1,Mitchell et al. 1945,42.44,7.06,paper,healthy
    protein,8,Baker 1969,56.2,4.2,figure,healthy
    protein,6,Thomas 1962,56.3,6.9,figure,comparison
    protein,8,Baker 1969,64.7,4.3,figure,healthy
    protein,8,Baker 1969,65.5,3.4,figure,healthy
    protein,5,Pawan & Clode 1960,68.3,2.2,figure,comparison
    protein,6,Thomas 1962,70.0,8.2,figure,comparison
    protein,9,Woodard 1986,70.0,11.8,figure,healthy
    protein,9,Woodard 1986,70.0,10.5,figure,healthy
    protein,8,Baker 1969,71.5,2.35,figure,healthy
    protein,2,Forbes et al. 1953,71.57,5.85,paper,healthy
    protein,8,Baker 1969,72.5,1.4,figure,healthy
    protein,8,Baker 1969,72.7,2.55,figure,healthy
    protein,8,Baker 1969,77.0,0.5,figure,healthy
    protein,8,Baker 1969,78.2,1.65,figure,healthy
    protein,3,Forbes et al. 1956,78.35,6.75,paper,healthy
    protein,8,Baker 1969,79.2,1.85,figure,healthy
    protein,8,Baker 1969,86.65,0.62,figure,healthy
    protein,8,Baker 1969,87.35,0.5,figure,healthy
    protein,8,Baker 1969,87.55,1.5,figure,healthy
    ash,1,Mitchell et al. 1945,42.44,0.51,paper,healthy
    ash,9,Woodard 1986,52.7,0.17,figure,healthy
    ash,9,Woodard 1986,69.5,0.05,figure,healthy
    ash,9,Woodard 1986,70.2,0.12,figure,healthy
    ash,2,Forbes et al. 1953,71.57,0.2,paper,healthy
    ash,3,Forbes et al. 1956,78.35,0.95,paper,healthy
    ash,9,Woodard 1986,79.8,0.095,figure,healthy
    ash,9,Woodard 1986,79.85,0.225,figure,healthy
    """
    function draw_wlp(seed, n; csv=IOBuffer(ADIPOSE_CSV))
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

# ╔═╡ aaaa0029-0000-4000-8000-000000000029
# Woodard & White 1986 Fig. 1, reproduced from the inlined ADIPOSE_CSV — this is the data
# `draw_wlp` fits its prior to, plotted so the table is visible without an external file.
# Colour = component, marker = provenance (● tabulated value, ✚ digitized from the figure).
let f=CM.Figure(size=(900,480))
    raw=readdlm(IOBuffer(ADIPOSE_CSV),','; header=false); hdr=string.(raw[1,:]); rows=raw[2:end,:]
    ci(x)=findfirst(==(x),hdr); cc,cl,cp,cv=ci("component"),ci("lipid_pct"),ci("component_pct"),ci("provenance")
    CWv=CM.RGBf(0.231,0.459,0.690); CPv=CM.RGBf(0.757,0.267,0.235); CAv=CM.RGBf(0.50,0.50,0.50)
    ax=CM.Axis(f[1,1];xlabel="lipid (mass %)",ylabel="component (mass %)",limits=(40,92,-2,56))
    CM.vlines!(ax,50;color=:gray,linestyle=:dash)
    CM.text!(ax,50.8,55;text="→ draw_wlp keeps lipid ≥ 50 %",color=:gray,fontsize=10,align=(:left,:top))
    for (comp,col) in (("water",CWv),("protein",CPv),("ash",CAv)), (prov,mk) in (("paper",:circle),("figure",:cross))
        s=[i for i in axes(rows,1) if string(rows[i,cc])==comp && string(rows[i,cv])==prov]
        isempty(s) && continue
        CM.scatter!(ax,[Float64(rows[i,cl]) for i in s],[Float64(rows[i,cp]) for i in s];color=col,marker=mk,markersize=10,label="$comp ($prov)")
    end
    CM.axislegend(ax;position=:rt,framevisible=false,labelsize=9)
    CM.Label(f[0,:],"Adipose composition vs lipid fraction — Woodard & White 1986 Fig. 1 · $(size(rows,1)) points, 7 studies · ● tabulated  ✚ digitized";fontsize=12,font=:bold)
    safe_save(joinpath(ASSET,"fig1_adipose_composition.png"),f); f
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
        (hu_lo=BS.synth_vmi_2basis(vwat,ciod;energy_keV=ELO),hu_hi=BS.synth_vmi_2basis(vwat,ciod;energy_keV=EHI),geom=slo.geom)
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
            v_lo=[Float64(acq.hu_lo[i,rc.midz]) for i in ci]; v_hi=[Float64(acq.hu_hi[i,rc.midz]) for i in ci]
            push!(out,(lab=lab,fw=c[1],fl=c[2],fp=c[3],v_lo=v_lo,v_hi=v_hi,m_lo=mean(v_lo),m_hi=mean(v_hi),s_lo=std(v_lo),s_hi=std(v_hi)))
        end
        (rois=out,midz=rc.midz,m2=rc.m2)
    end
    md"`build_thorax` (packed / sector / centred) · `run_acq` (1300-col, bowtie-free, 984-view) · `roi_cores`/`collect_rois`"
end

# ╔═╡ aaaa0011-0000-4000-8000-000000000011
md"## 5 · Inverse: calibration surface · noise · edge-preserving TV"

# ╔═╡ aaaa0012-0000-4000-8000-000000000012
begin
    poly2(hl,hh)=[1.0,hl,hh,hl^2,hh^2,hl*hh]; surf(c,hl,hh)=dot(c,poly2(hl,hh))
    quad_sigma(c,H)=c[1]*H^2+c[2]*H+c[3]
    fit_sigma_quad(hu,sig)=(X=hcat(hu.^2,hu,ones(length(hu)));c=X\sig;c[1]<0&&(Xa=hcat(hu,ones(length(hu)));ca=Xa\sig;c=[0.0,ca[1],ca[2]]);c)
    function metrics(t,r)
        mt,mr=mean(t),mean(r);st2=mean((t.-mt).^2);sr2=mean((r.-mr).^2);str=mean((t.-mt).*(r.-mr))
        (ccc=2str/(st2+sr2+(mt-mr)^2),slope=str/st2,int=mr-str/st2*mt,rmse=sqrt(mean((r.-t).^2)),r2=1-sum((r.-t).^2)/sum((t.-mt).^2))
    end
    # coupled edge-preserving Huber-TV on (f_l,f_p); w = optional σ_f data weight (1/σ_f²). Never Gaussian.
    # λ is ABSOLUTE, weighed against w=1/σ_f² in `den` — so it must be O(w), not O(1). See aaaa0031
    # for the measurement that sets it; noisier data ⇒ smaller w ⇒ TV self-strengthens (that is the point).
    # NOTE λ has no default and no const: it is FITTED from calibration data (§6, fit_tv_lambda), so it
    # cannot be known here. Passing it explicitly is the point — a default would be a second, stale home
    # for a number the data owns. TV_ITERS/TV_EPS/TV_SIMPLEX are genuine choices, so they are consts.
    const TV_ITERS = 25; const TV_EPS = 0.04
    const TV_SIMPLEX = :once     # project the RESULT, never per sweep — per-sweep rectification is a
                                 # Jensen bias on any region mean drawn from the map (measured: §6.5)
    # `simplex` = when to project onto {f≥0, f_l+f_p≤1}. :each rectifies every sweep — which is the
    # per-voxel rectification this notebook refuses in the scoring path ("would bias the ROI mean
    # (Jensen)"), applied `iters` times. :once projects only the result; :never leaves it raw.
    function tv_coupled(yl,yp,mask; lambda,iters=TV_ITERS,eps=TV_EPS,w=nothing,simplex=TV_SIMPLEX)
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
                fl2[i,j],fp2[i,j] = simplex===:each ? smp(rl/den,rp/den) : (rl/den,rp/den)
            end
            fl,fl2=fl2,fl;fp,fp2=fp2,fp
        end
        simplex===:once && (@inbounds for j in 1:ny,i in 1:nx
            mask[i,j] && ((fl[i,j],fp[i,j])=smp(fl[i,j],fp[i,j])); end)
        ([mask[i,j] ? fl[i,j] : NaN for i in 1:nx,j in 1:ny],[mask[i,j] ? fp[i,j] : NaN for i in 1:nx,j in 1:ny])
    end
    dpoly_lo(hl,hh)=[0.0,1.0,0.0,2hl,0.0,hh]; dpoly_hi(hl,hh)=[0.0,0.0,1.0,0.0,2hh,hl]
    label_centroid(m2,lab)=(idx=findall(==(UInt8(lab)),m2); (mean(getindex.(idx,1)),mean(getindex.(idx,2))))
    struct BayesPrior; μ_w::Float64; s_w::Float64; μ_l::Float64; s_l::Float64; α_p::Float64; θ_p::Float64; end
    bayes_prior_broad(comps;s_wl=0.15,fp_shape=1.2,fp_scale=0.10)=(fw=[c[1] for c in comps];fl=[c[2] for c in comps];BayesPrior(mean(fw),s_wl,mean(fl),s_wl,fp_shape,fp_scale))
    md"`surf` (calibration) · `fit_sigma_quad` · `metrics` (CCC…) · `tv_coupled` (σ\_f Huber-TV) · `bayes_prior_broad`"
end

# ╔═╡ aaaa0013-0000-4000-8000-000000000013
md"""## 6 · Run sims + calibrate + decode (cached)
4 calibration + 1 map thorax; **5 circular-test + 4 sector-test** held-out (n = 65 + 64 = 129 ROIs); 4 centred
integrated-HU sims. First run ≈ 20 min on GPU; results cache to `data/wlp_*_cache_70_150.jls`. Delete those to re-sim."""

# ╔═╡ aaaa0014-0000-4000-8000-000000000014
begin
    const PTAG   = "$(Int(ELO))_$(Int(EHI))"                       # pair-derived cache tag (each keV pair its own cache)
    const CACHE  = joinpath(DATA, "wlp_sim_cache_$(PTAG).jls")     # calibration + delivered-map thorax
    const TCACHE = joinpath(DATA, "wlp_test_cache_$(PTAG).jls")    # circular held-out test (5 sims)
    const ICACHE = joinpath(DATA, "wlp_int_cache_$(PTAG).jls")
    const SCACHE = joinpath(DATA, "wlp_sect_cache_$(PTAG).jls")
    const CORE_RPX = round(Int, INS_R/RECON_PX_MM - EROSION_PX)   # eroded interior core ≈ 225 mm²
    const IFL = 0.85
    # one packed-thorax acquisition → NHEART eroded insert cores (shared by calibration + circular test)
    _cores(seed, cseed) = begin
        comps = diverse_comps(cseed, NHEART); ph = build_thorax(comps); acq = run_acq(ph.gpu; seed=seed)
        lc = Dict(ROD0-1+k => comps[k] for k in 1:NHEART); r = collect_rois(acq, ph.cpu, lc; radius_px=CORE_RPX).rois
        img = (; seed, hu_lo=Array(acq.hu_lo), hu_hi=Array(acq.hu_hi), comps)   # full z-stack CT for raw export (shared geometry ⇒ map_m2 labels)
        ph=nothing; GC.gc(true); (rois=r, img=img)
    end
    # ── calibration (4 sims) + delivered-map thorax ──
    if !isfile(CACHE)
        println("── running calibration + map thorax ──"); Random.seed!(1)
        calrois = NamedTuple[]; calsims = NamedTuple[]
        for (si,seed) in enumerate((11,12,13,14)); c=_cores(seed, 1000+si); append!(calrois, c.rois); push!(calsims, c.img); end
        mcomps = diverse_comps(777, NHEART); mph = build_thorax(mcomps); macq = run_acq(mph.gpu; seed=999)
        mrc = roi_cores(mph.cpu, macq.geom, (RECON_N,RECON_N,3), collect(ROD0:(ROD0+NHEART-1)))
        map_lo_v = Array(macq.hu_lo); map_hi_v = Array(macq.hu_hi); map_m2 = mrc.m2; mph=nothing; GC.gc(true)   # full z-stack; 2D map derived below
        serialize(CACHE, (; calrois, map_lo_v, map_hi_v, map_m2, mcomps, calsims))
    end
    Dm = deserialize(CACHE); calrois=Dm.calrois; map_lo_v,map_hi_v,map_m2,mcomps = Dm.map_lo_v,Dm.map_hi_v,Dm.map_m2,Dm.mcomps; calsims=Dm.calsims
    mmid = size(map_hi_v,3)÷2+1; map_lo = map_lo_v[:,:,mmid]; map_hi = map_hi_v[:,:,mmid]   # notebook figures/decode use the 2D mid slice
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
            im = size(ia.hu_hi,3)÷2+1; m2 = roi_cores(ip.cpu, ia.geom, (RECON_N,RECON_N,3), [ROD0]).m2
            push!(isims, (r=r, hu_lo=Array(ia.hu_lo[:,:,im]), hu_hi=Array(ia.hu_hi[:,:,im]), m2=m2)); ip=nothing; GC.gc(true)
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
            push!(sectsims, (; seed, hu_lo=Array(sa.hu_lo), hu_hi=Array(sa.hu_hi), comps=sc)); sp=nothing; GC.gc(true)   # full z-stack
        end
        scomps = diverse_comps(70260800, NSECT); smph = build_thorax(scomps; sectors=(SECT_NANG,SECT_NRAD)); smacq = run_acq(smph.gpu; seed=403)
        smrc = roi_cores(smph.cpu, smacq.geom, (RECON_N,RECON_N,3), collect(ROD0:(ROD0+NSECT-1)))
        smap_lo_v = Array(smacq.hu_lo); smap_hi_v = Array(smacq.hu_hi); smap_m2 = smrc.m2; smph=nothing; GC.gc(true)   # full z-stack; 2D map derived below
        serialize(SCACHE, (; sectrois, smap_lo_v, smap_hi_v, smap_m2, scomps, sectsims))
    end
    DS = deserialize(SCACHE); sectrois=DS.sectrois; smap_lo_v,smap_hi_v,smap_m2,scomps = DS.smap_lo_v,DS.smap_hi_v,DS.smap_m2,DS.scomps; sectsims=DS.sectsims
    smid = size(smap_hi_v,3)÷2+1; smap_lo = smap_lo_v[:,:,smid]; smap_hi = smap_hi_v[:,:,smid]   # notebook figures/decode use the 2D mid slice

    # ── calibration: quadratic surface (point accuracy) + AFFINE lipid (integrals) + noise ladder ──
    m_lo_cal=[r.m_lo for r in calrois]; m_hi_cal=[r.m_hi for r in calrois]
    fwc=[r.fw for r in calrois]; flc=[r.fl for r in calrois]; fpc=[r.fp for r in calrois]
    Xc=reduce(vcat,[poly2(m_lo_cal[i],m_hi_cal[i])' for i in eachindex(m_lo_cal)])
    cw=Xc\fwc; cl=Xc\flc; cp=Xc\fpc                                       # unconstrained LS — pinning to theoretical corners bent near-pure-fat 3× (2026-07-14)
    Xa=hcat(ones(length(m_lo_cal)),m_lo_cal,m_hi_cal); cl_aff=Xa\flc                 # affine f_l — linear ⇒ commutes with the PSF
    r2fit(c,y)=1-sum((surf.(Ref(c),m_lo_cal,m_hi_cal).-y).^2)/sum((y.-mean(y)).^2)
    sc_lo=fit_sigma_quad(m_lo_cal,[r.s_lo for r in calrois]); sc_hi=fit_sigma_quad(m_hi_cal,[r.s_hi for r in calrois])
    res_lo=vcat([r.v_lo.-r.m_lo for r in calrois]...); res_hi=vcat([r.v_hi.-r.m_hi for r in calrois]...); ρ=cor(res_lo,res_hi)
    adipose=draw_wlp(20260708,2000); prior=bayes_prior_broad(adipose)
    decode(a,b)=(x=surf(cw,a,b);y=surf(cl,a,b);z=surf(cp,a,b);s=x+y+z;(x/s,y/s,z/s))
    aff_l(a,b)=cl_aff[1]+cl_aff[2]*a+cl_aff[3]*b

    # ── out-of-triangle handling: noise-ellipse (Mahalanobis) MLE projection onto the simplex ──
    # The raw surface decode divides by the sum, so a noisy HU can decode to an INFEASIBLE
    # (negative) composition. The maximum-likelihood feasible composition is the point on the
    # decomposition triangle closest to the measurement in the noise metric Σ⁻¹ (not Euclidean:
    # σ₇₀≫σ₁₅₀ and ρ≈0.79, so the noise ball is a tilted ellipse). G maps (f_l,f_p) → measured HU;
    # bias_hu = mean(recon − linear-mix theory) over the cal rods de-biases the exact inverse.
    Gmat = [PL[1]-PW[1] PP[1]-PW[1]; PL[2]-PW[2] PP[2]-PW[2]]
    Σhu  = (s70=std(res_lo); s150=std(res_hi); [s70^2 ρ*s70*s150; ρ*s70*s150 s150^2])
    Σinv = inv(Σhu); Amet = Gmat' * Σinv * Gmat                    # metric on (f_l,f_p)
    bias_hu = [mean(m_lo_cal .- [r.fl*PL[1]+r.fp*PP[1] for r in calrois]),
               mean(m_hi_cal .- [r.fl*PL[2]+r.fp*PP[2] for r in calrois])]
    insimplex(fl,fp) = fl≥-1e-9 && fp≥-1e-9 && (fl+fp)≤1+1e-9
    function proj_simplex(θ)                                        # θ=(f_l,f_p); Mahalanobis-closest vertex/edge
        insimplex(θ...) && return θ
        V=([0.0,0.0],[1.0,0.0],[0.0,1.0]); best=V[1]; bd=Inf        # water, lipid, protein corners
        for (p,q) in ((V[1],V[2]),(V[1],V[3]),(V[2],V[3]))
            u=q.-p; t=clamp(dot(u,Amet*(collect(θ).-p))/dot(u,Amet*u),0.0,1.0); c=p.+t.*u
            dv=c.-collect(θ); d=dot(dv,Amet*dv); d<bd && (bd=d; best=c)
        end
        (best[1],best[2])
    end
    # feasible decode: interior → validated surface; exterior → bias-corrected G-inverse, MLE-projected
    function decode_feas(a,b)
        r=(surf(cw,a,b),surf(cl,a,b),surf(cp,a,b))
        all(r.≥-1e-9) && (s=sum(r); return (r[1]/s,r[2]/s,r[3]/s))
        θ=Gmat\([a,b].-bias_hu); (fl,fp)=proj_simplex((θ[1],θ[2])); (1-fl-fp,fl,fp)
    end

    # ── cached-ROI diagnostics ONLY (out-of-triangle rate + pool-then-decode). The SCORED numbers are
    # NOT here: they come from the delivered estimator via score_rois below, once fullfield/
    # sigma_f_weight exist. One estimator feeds fig5, the §8 table and the model TOML together.
    allrois=vcat(testrois,sectrois)
    tfw_c=[r.fw for r in allrois]; tfl_c=[r.fl for r in allrois]; tfp_c=[r.fp for r in allrois]
    pfl_pool=[decode(r.m_lo,r.m_hi)[2] for r in allrois]; ml_pool=metrics(tfl_c,pfl_pool)

    # ── out-of-triangle diagnostic + feasible pooled decode (noise-ellipse MLE) ──
    # Per-voxel decodes leave the simplex under noise; that is EXPECTED (a near-edge composition
    # scattered by ε). We do NOT project per-voxel — rectifying each voxel before averaging would
    # bias the ROI mean (Jensen). We report the per-voxel infeasible rate, then deliver the pooled
    # ROI decode through decode_feas so any ROI whose MEAN still lands outside is MLE-projected.
    # TV_SIMPLEX=:once obeys the same rule inside the map: :each rectified on all TV_ITERS sweeps and
    # cost 0.03 of f_l slope — the very Jensen bias this paragraph refuses. See the clamp test.
    nvox_all=sum(length(r.v_lo) for r in allrois)
    nvox_out=sum(count(any(decode(r.v_lo[j],r.v_hi[j]).<-1e-6) for j in eachindex(r.v_lo)) for r in allrois)
    nroi_out=count(any(decode(r.m_lo,r.m_hi).<-1e-6) for r in allrois)
    pfeas=[decode_feas(r.m_lo,r.m_hi) for r in allrois]                 # feasible-by-construction ROI composition
    pfw_feas=getindex.(pfeas,1); pfl_feas=getindex.(pfeas,2); pfp_feas=getindex.(pfeas,3)
    mw_feas=metrics(tfw_c,pfw_feas); ml_feas=metrics(tfl_c,pfl_feas); mp_feas=metrics(tfp_c,pfp_feas)

    # ── delivered map: per-voxel decode over gated soft tissue + σ_f-weighted edge-preserving Huber-TV ──
    const SOFT_HU_LO, SOFT_HU_HI = -300.0, 250.0
    function sigma_f_weight(h_lo,h_hi)
        w=fill(NaN,size(h_hi))
        for I in CartesianIndices(h_hi); (SOFT_HU_LO<h_hi[I]<SOFT_HU_HI)||continue
            hl=Float64(h_lo[I]);hh=Float64(h_hi[I]); sig_lo=quad_sigma(sc_lo,hl);sig_hi=quad_sigma(sc_hi,hh)
            g_lo_l=dot(cl,dpoly_lo(hl,hh));g_hi_l=dot(cl,dpoly_hi(hl,hh)); g_lo_p=dot(cp,dpoly_lo(hl,hh));g_hi_p=dot(cp,dpoly_hi(hl,hh))
            vl=g_lo_l^2*sig_lo^2+g_hi_l^2*sig_hi^2+2ρ*g_lo_l*g_hi_l*sig_lo*sig_hi; vp=g_lo_p^2*sig_lo^2+g_hi_p^2*sig_hi^2+2ρ*g_lo_p*g_hi_p*sig_lo*sig_hi
            w[I]=1.0/max(vl+vp,1e-6)
        end; w
    end
    function fullfield(h_lo,h_hi)
        fw=fill(NaN,size(h_hi));fl=copy(fw);fp=copy(fw)
        for I in CartesianIndices(h_hi); (SOFT_HU_LO<h_hi[I]<SOFT_HU_HI)||continue
            d=decode(Float64(h_lo[I]),Float64(h_hi[I])); fw[I]=d[1];fl[I]=d[2];fp[I]=d[3]; end
        (fw,fl,fp)
    end
    # ── λ is FITTED, on CALIBRATION data only ────────────────────────────────────────────────────
    # The held-out scans (circular test + sector) must never inform λ, or "held-out" is a fiction and
    # every number in §8 is a training score. The 4 calibration thoraxes already carry GT comps and
    # full images, so the fit has everything it needs without touching them.
    #
    # Objective: per-voxel RMSE vs GT over the calibration cores — the exact quantity TV exists to
    # reduce. Minimised by golden-section on log₁₀λ: the curve is unimodal in log λ (noise-limited on
    # the left, bias-limited on the right), there is no gradient, and a bracket search needs no
    # dependency. λ therefore has no literal anywhere — move the scanner or the keV pair and it refits.
    core_idx(m2,lab,rpx)=(idx=findall(==(UInt8(lab)),m2); isempty(idx) ? CartesianIndex{2}[] :
        (cx=mean(getindex.(idx,1));cy=mean(getindex.(idx,2)); [I for I in idx if (I[1]-cx)^2+(I[2]-cy)^2≤rpx^2]))
    _midz(v)=size(v,3)÷2+1
    CALPREP=[(img=s, m2=map_m2, rpx=CORE_RPX, nins=NHEART,          # calibration scans share HEARTC geometry
              P=(f0=fullfield(s.hu_lo[:,:,_midz(s.hu_lo)],s.hu_hi[:,:,_midz(s.hu_hi)]),
                 w=sigma_f_weight(s.hu_lo[:,:,_midz(s.hu_lo)],s.hu_hi[:,:,_midz(s.hu_hi)])))
             for s in calsims]                                       # decode once; each λ only re-runs TV
    function cal_pv_rmse(lam; simplex=TV_SIMPLEX)                    # per-voxel RMSE vs GT, CALIBRATION only
        t=Float64[]; r=Float64[]
        for s in CALPREP
            gate=.!isnan.(s.P.f0[2])
            fl,fp = lam≤0 ? (s.P.f0[2],s.P.f0[3]) : tv_coupled(s.P.f0[2],s.P.f0[3],gate; lambda=lam,w=s.P.w,simplex=simplex)
            F=(map((a,b)-> isnan(a) ? NaN : 1-a-b, fl,fp), fl, fp)
            for k in 1:s.nins, I in core_idx(s.m2,ROD0-1+k,s.rpx), c in 1:3
                isfinite(F[c][I]) || continue
                push!(t, s.img.comps[k][c]); push!(r, F[c][I])
            end
        end
        sqrt(mean((r.-t).^2))
    end
    function golden_min(f,lo,hi; tol=0.02)                           # 1-D min of a unimodal f on [lo,hi]
        φ=(sqrt(5)-1)/2; a,b=lo,hi
        c=b-φ*(b-a); d=a+φ*(b-a); fc=f(c); fd=f(d); n=2
        while (b-a)>tol
            if fc<fd; b,d,fd=d,c,fc; c=b-φ*(b-a); fc=f(c)
            else;     a,c,fc=c,d,fd; d=a+φ*(b-a); fd=f(d); end
            n+=1
        end
        x=(a+b)/2; (x=x, f=f(x), n=n+1)
    end
    _gold = golden_min(u->cal_pv_rmse(10.0^u), -1.0, 2.0; tol=0.02)  # λ∈[0.1,100]; tol ⇒ λ to ~5%
    TV_LAMBDA = round(10.0^_gold.x, digits=2)                        # fitted — no literal, no const
    # unimodality is an assumption of golden-section, so check the bracket rather than trust it
    _uni_ok = cal_pv_rmse(TV_LAMBDA) ≤ min(cal_pv_rmse(TV_LAMBDA/3), cal_pv_rmse(TV_LAMBDA*3)) + 1e-9

    function deliver(m_lo,m_hi,m2,comps; lambda=TV_LAMBDA, simplex=TV_SIMPLEX)
        f0=fullfield(m_lo,m_hi); gate=.!isnan.(f0[2]); w=sigma_f_weight(m_lo,m_hi)
        fl_tv,fp_tv=tv_coupled(f0[2],f0[3],gate; lambda=lambda,w=w,simplex=simplex)
        fw_tv=map((a,b)-> isnan(a) ? NaN : 1-a-b, fl_tv, fp_tv)
        rec=cat(fw_tv,fl_tv,fp_tv;dims=3); tru=fill(NaN,size(m2)...,3); recgt=fill(NaN,size(m2)...,3)
        for k in 1:length(comps); lab=ROD0-1+k
            idx=findall(==(UInt8(lab)),m2); isempty(idx)&&continue
            d=decode(mean(Float64(m_lo[I]) for I in idx),mean(Float64(m_hi[I]) for I in idx))
            for I in idx; tru[I,1],tru[I,2],tru[I,3]=comps[k]; recgt[I,1],recgt[I,2],recgt[I,3]=d; end
        end
        (rec=rec,tru=tru,recgt=recgt)
    end
    circ = deliver(map_lo,map_hi,map_m2,mcomps)
    sect = deliver(smap_lo,smap_hi,smap_m2,scomps)
    recmap=circ.rec; truemap=circ.tru; recmap_gt=circ.recgt

    # ── the scored estimator IS the delivered one ────────────────────────────────────────────────
    # One enumeration of the held-out cores, run through the same chain the map ships (per-voxel
    # decode → σ_f Huber-TV → simplex). fig5, the §8 table and wlp_model_*.toml all read mw/ml/mp,
    # so TV_LAMBDA/TV_SIMPLEX move plot, table and snapshot together — nothing is transcribed.
    # Every scan reuses its geometry's label map (only comps differ per scan), so the cores are
    # recoverable from the stored images; the asserts below pin this to the cached n.
    # These are REPORTED, never optimised against — λ was already fixed above, on calibration.
    TESTSIMS=vcat([(img=s,m2=map_m2, rpx=CORE_RPX,nins=NHEART,geom=:circular) for s in testsims],
                  [(img=s,m2=smap_m2,rpx=7,       nins=NSECT, geom=:sector)   for s in sectsims])
    function score_rois(; lambda=TV_LAMBDA, simplex=TV_SIMPLEX)
        out=NamedTuple[]
        for s in TESTSIMS
            m_lo=s.img.hu_lo[:,:,_midz(s.img.hu_lo)]; m_hi=s.img.hu_hi[:,:,_midz(s.img.hu_hi)]
            f0=fullfield(m_lo,m_hi); gate=.!isnan.(f0[2]); w=sigma_f_weight(m_lo,m_hi)
            fl,fp = lambda≤0 ? (f0[2],f0[3]) : tv_coupled(f0[2],f0[3],gate; lambda=lambda,w=w,simplex=simplex)
            D=(map((a,b)-> isnan(a) ? NaN : 1-a-b, fl,fp), fl, fp)
            for k in 1:s.nins
                ci=core_idx(s.m2,ROD0-1+k,s.rpx); isempty(ci)&&continue
                v=[Float64[D[c][I] for I in ci if isfinite(D[c][I])] for c in 1:3]
                any(isempty,v) && continue
                push!(out,(geom=s.geom, t=s.img.comps[k], p=ntuple(c->mean(v[c]),3),
                           sem=ntuple(c->std(v[c])/sqrt(length(v[c])),3), nvox=length(v[1])))
            end
        end
        out
    end
    drois=score_rois()
    @assert length(drois)==length(allrois) "scorer found $(length(drois)) ROIs; cached rois have $(length(allrois))"
    @assert sort([r.t[2] for r in drois])≈sort(tfl_c) "scorer and cached rois disagree on the truth set"
    geomtag=[r.geom for r in drois]
    tfw=[r.t[1] for r in drois]; tfl=[r.t[2] for r in drois]; tfp=[r.t[3] for r in drois]
    pfw=[r.p[1] for r in drois]; pfl=[r.p[2] for r in drois]; pfp=[r.p[3] for r in drois]
    semfw=[r.sem[1] for r in drois]; semfl=[r.sem[2] for r in drois]; semfp=[r.sem[3] for r in drois]
    mw=metrics(tfw,pfw); ml=metrics(tfl,pfl); mp=metrics(tfp,pfp)
    dHU_hi=[abs(mix_hu(pfw[i],pfl[i],pfp[i],EHI)-mix_hu(tfw[i],tfl[i],tfp[i],EHI)) for i in eachindex(tfw)]

    # ── integrated-HU: EXCESS lipid over local muscle; conservation recovers ∫(f_l−bg) without the boundary ──
    const FIXED_MARGIN_PX = 8.0
    IFAT=(1-IFL-0.05,IFL,0.05); FAT_AFF=aff_l(mix_hu(IFAT...,ELO),mix_hu(IFAT...,EHI))   # affine content; vs IFL = decode bias
    integ = NamedTuple[]
    for s in isims
        i_lo=s.hu_lo; i_hi=s.hu_hi; m2=s.m2; r_mm=s.r; rpx=r_mm/RECON_PX_MM; (cx,cy)=label_centroid(m2,ROD0)
        qb(i,j)=[1.0,i-cx,j-cy,(i-cx)^2,(j-cy)^2,(i-cx)*(j-cy)]                          # global quadratic muscle bg (cupping)
        hm=[CartesianIndex(i,j) for i in axes(i_lo,1),j in axes(i_lo,2) if m2[i,j]==0x02 && (i-cx)^2+(j-cy)^2≤(60.0/RECON_PX_MM)^2]
        cg=reduce(vcat,[qb(I[1],I[2])' for I in hm])\[aff_l(Float64(i_lo[I]),Float64(i_hi[I])) for I in hm]
        bgf(i,j)=dot(cg,qb(i,j)); bg0=bgf(cx,cy); truelip=π*r_mm^2*(FAT_AFF-bg0)
        margins=0.0:1.0:16.0; recov=Float64[]
        for mg in margins
            R=rpx+mg; acc=0.0
            for i in axes(i_lo,1), j in axes(i_lo,2)
                (i-cx)^2+(j-cy)^2≤R^2 || continue; (SOFT_HU_LO<i_hi[i,j]<SOFT_HU_HI) || continue
                acc += aff_l(Float64(i_lo[i,j]),Float64(i_hi[i,j])) - bgf(i,j)
            end
            push!(recov, acc*RECON_PX_MM^2)
        end
        bi=findfirst(==(FIXED_MARGIN_PX),margins)
        push!(integ,(r=r_mm,fl=IFL,truelip=truelip,bg=bg0,naivelip=recov[1],intlip=recov[bi],margins=collect(margins),recov=recov))
    end
    # Accuracy (CCC/slope/RMSE), cond(G) and the integrated-HU recovery all live in §8 — this cell
    # reports only what §8 does not: the calibration fit and the noise/feasibility diagnostics.
    Markdown.parse("""
**Calibration** — n=$(length(calrois)) rods, R²(f\\_w surface fit)=**$(round(r2fit(cw,fwc),digits=3))**, inter-energy noise correlation ρ=**$(round(ρ,digits=2))**.

**Out-of-triangle** — $(round(100nvox_out/nvox_all,digits=1))% of per-voxel decodes ($(nvox_out)/$(nvox_all)) land outside the W/L/P simplex, as expected when noise scatters a near-edge composition; they are *not* rectified per-voxel. Only $(nroi_out)/$(length(allrois)) ROI *means* land outside, and those go through the noise-ellipse MLE projection.

**Held-out test** — n=$(length(drois)) ($(count(==(:circular),geomtag)) circular + $(count(==(:sector),geomtag)) sector), scored in §8 through the delivered estimator: per-voxel decode + σ\\_f Huber-TV, λ=$(TV_LAMBDA), simplex=`:$(TV_SIMPLEX)`.
""")
end

# ╔═╡ aaaa0030-0000-4000-8000-000000000030
# The phantom label map as every downstream figure actually sees it: `map_m2` / `smap_m2`, i.e. the
# 0.4 mm mask nearest-neighbour resampled to the 512² recon grid, mid-slice. Axes are phantom-frame
# mm (not pixels) so the geometry is readable directly; yreversed ⇒ spine down = standard CT view.
# Left: 13 hex-packed circular inserts (calibration + circular test). Right: 16 sector wedges, the
# held-out test geometry. Insert numbers k index `mcomps[k]` / `scomps[k]` — label = ROD0-1+k.
#
# Insert colour is a deliberately neutral "mixture slot", NOT a tissue colour: the inserts are
# `diverse_comps` spanning the whole W/L/P triangle (f_w 0.02–0.83 here — over half are water-
# dominant), so they are not adipose and no tissue colour would be honest. Amber/blue/red are
# reserved for the lipid/water/protein endpoints in fig_decode_triangle_noise. The label map is
# also shared across scans while each scan redraws its own comps, so colour cannot encode
# composition here — see the delivered map (fig3/fig7) for that.
let f=CM.Figure(size=(1320,600))
    x_mm(i)=185.0+(-RECON_FOV_MM/2+RECON_PX_MM/2+(i-1)*RECON_PX_MM)   # recon px → phantom mm (auto-centred on isocentre)
    y_mm(j)=135.0+(-RECON_FOV_MM/2+RECON_PX_MM/2+(j-1)*RECON_PX_MM)
    TIS=(0x00=>("air",CM.RGBf(1.00,1.00,1.00)), 0x01=>("lung",CM.RGBf(0.78,0.86,0.93)),
         0x02=>("muscle",CM.RGBf(0.75,0.44,0.42)), 0x03=>("cortical bone",CM.RGBf(0.93,0.91,0.83)),
         0x04=>("red marrow",CM.RGBf(0.85,0.55,0.58)), 0x05=>("adipose (fat ring)",CM.RGBf(0.97,0.83,0.46)))
    INS=CM.RGBf(0.42,0.38,0.60); cmap=Dict(k=>c for (k,(_,c)) in TIS)
    colorize(m)=[get(cmap,l,INS) for l in m]                          # anything ≥ ROD0 is a WLP insert
    xr=(185.0-RECON_FOV_MM/2,185.0+RECON_FOV_MM/2); yr=(135.0-RECON_FOV_MM/2,135.0+RECON_FOV_MM/2)  # image! wants outer edges
    θ=range(0,2π,200)
    # Circular inserts are compact ⇒ centroid is fine. Sector wedges are annular, so their centroid
    # drifts inward and the inner ring collides at the hub: place those on the wedge bisector at a
    # fixed fraction of the ring, derived from the same constants that build them.
    cpos(k)=(t=label_centroid(map_m2,UInt8(ROD0-1+k)); (x_mm(t[1]),y_mm(t[2])))
    spos(k)=(a=((k-1)%SECT_NANG+0.5)*(2π/SECT_NANG); r=(SECT_R_MM/SECT_NRAD)*((k-1)÷SECT_NANG+0.55); (HC_X+r*cos(a),HC_Y+r*sin(a)))
    for (col,(m,cmps,pos,ttl)) in enumerate(((map_m2,mcomps,cpos,"circular — $(length(mcomps)) hex-packed inserts, ø$(round(2*INS_R,digits=1)) mm"),
                                             (smap_m2,scomps,spos,"sector — $(length(scomps)) wedges, r ≤ $(Int(SECT_R_MM)) mm (held-out)")))
        ax=CM.Axis(f[1,col];aspect=CM.DataAspect(),yreversed=true,xlabel="x (mm)",ylabel=col==1 ? "y (mm)" : "",title=ttl,titlesize=11)
        CM.image!(ax,xr,yr,colorize(m))
        CM.lines!(ax,HC_X.+HEART_R_MM.*cos.(θ),HC_Y.+HEART_R_MM.*sin.(θ);color=:black,linestyle=:dash,linewidth=1.1)
        CM.text!(ax,HC_X+HEART_R_MM+6,HC_Y;text="heart cavity\nr = $(Int(HEART_R_MM)) mm",fontsize=9,align=(:left,:center))
        if col==2                       # wedges share one colour ⇒ draw the lattice or it reads as a solid disc
            for a in 0:SECT_NANG-1
                φ=a*(2π/SECT_NANG); CM.lines!(ax,[HC_X,HC_X+SECT_R_MM*cos(φ)],[HC_Y,HC_Y+SECT_R_MM*sin(φ)];color=(:white,0.8),linewidth=0.8)
            end
            for ri in 1:SECT_NRAD-1
                rr=ri*(SECT_R_MM/SECT_NRAD); CM.lines!(ax,HC_X.+rr.*cos.(θ),HC_Y.+rr.*sin.(θ);color=(:white,0.8),linewidth=0.8)
            end
        end
        for k in 1:length(cmps)
            px,py=pos(k); isnan(px) && continue
            CM.text!(ax,px,py;text=string(k),color=:white,fontsize=9,font=:bold,align=(:center,:center))
        end
    end
    swatch(c)=CM.PolyElement(color=c,strokecolor=CM.RGBf(0.6,0.6,0.6),strokewidth=0.5)   # stroke: air is white-on-white
    els=CM.PolyElement[]; lbls=String[]
    for (k,(n,c)) in TIS; push!(els,swatch(c)); push!(lbls,"$(Int(k)) · $n"); end
    push!(els,swatch(INS)); push!(lbls,"$(ROD0)+ · W/L/P mixture\n        slots (per-scan comps)")
    CM.Legend(f[1,3],els,lbls,"label → material";framevisible=false,labelsize=10,titlesize=10)
    CM.Label(f[0,:],"QRM-thorax phantom label map — 512² recon grid, mid-slice, phantom-frame mm · posterior down · numbers index mcomps/scomps";fontsize=12,font=:bold)
    safe_save(joinpath(ASSET,"fig_phantom_labels.png"),f); f
end

# ╔═╡ aaaa0031-0000-4000-8000-000000000031
# ── Noise measurement + the λ fit, ON CALIBRATION. Nothing here is chosen by how smooth it looks,
# and nothing here touches the held-out scans: §6 fits λ from calibration, this cell shows the work.
begin
    # (1) WHAT distribution? Measure, don't assume. Poisson lives in the PROJECTION domain; each FBP
    # voxel is a filtered weighted sum of ~10³ rays, so the CLT flattens the marginal back to Gaussian.
    # The Poisson origin survives only as the σ(HU) ladder — the variance, not the shape.
    _mom(v)=(m=mean(v);s=std(v);(sd=s,skew=mean(((v.-m)./s).^3),exkurt=mean(((v.-m)./s).^4)-3))
    nz_lo=_mom(res_lo); nz_hi=_mom(res_hi)          # res_* are calibration-rod residuals

    # (2) Is it iid? A per-voxel 1/σ² weight silently assumes yes. FBP streaks say otherwise, and a
    # correlated noise field is exactly what a per-voxel weight CANNOT see. (core_idx: from §6.)
    _cal1 = calsims[1]                              # a CALIBRATION scan, not the display/test thoraxes
    _cal1_hi = _cal1.hu_hi[:,:,_midz(_cal1.hu_hi)]
    function resid_img(h,m2,nins,rpx)               # HU − per-core mean, on eroded cores only (no PV edges)
        r=fill(NaN,size(h))
        for k in 1:nins; ci=core_idx(m2,ROD0-1+k,rpx); isempty(ci)&&continue
            mu=mean(Float64(h[I]) for I in ci); for I in ci; r[I]=Float64(h[I])-mu; end; end
        r
    end
    function acf1d(r,maxlag,dim)
        s2=mean(x^2 for x in r if isfinite(x)); out=Float64[]
        for d in 0:maxlag
            acc=0.0;n=0
            for J in CartesianIndices(r)
                K=dim==1 ? CartesianIndex(J[1]+d,J[2]) : CartesianIndex(J[1],J[2]+d)
                checkbounds(Bool,r,K)||continue; (isfinite(r[J])&&isfinite(r[K]))||continue
                acc+=r[J]*r[K];n+=1
            end
            push!(out,acc/max(n,1)/s2)
        end; out
    end
    rimg_hi=resid_img(_cal1_hi,map_m2,NHEART,CORE_RPX)
    ACF_LAG=6; acf_x=acf1d(rimg_hi,ACF_LAG,1); acf_y=acf1d(rimg_hi,ACF_LAG,2)
    acf_len=1+2*sum(max.(acf_x[2:end],0.0))         # ≈ #voxels per independent sample along x

    # (3) The scale check: `den = w + Σ_nbr c`, w = 1/σ_f² (O(10²)), c = λ/max(‖∇f‖,eps) ≤ λ/eps.
    # In a flat region ‖∇f‖ floors at eps, so the 4 neighbours together weigh 4λ/eps against w.
    w_med=median(x for x in CALPREP[1].P.w if isfinite(x))
    tv_pull(lam)=4*(lam/TV_EPS)/w_med               # TV:data leverage in a flat region. ≪1 ⇒ TV is decorative.

    # (4) The λ fit, shown. This is the SAME objective §6 minimised (cal_pv_rmse), evaluated on a grid
    # purely to display the curve golden-section walked — the grid selects nothing.
    LAMS=[0.0,0.05,0.3,1.0,3.0,5.0,10.0,15.0,20.0,30.0,50.0,100.0]
    cal_curve=[cal_pv_rmse(l) for l in LAMS]
    _tag(l)= l==TV_LAMBDA ? " ← **fitted λ**" : ""
    _rows=join(["| $(l==0 ? "0 (raw)" : string(l)) | $(round(cal_curve[i],digits=4)) |$(_tag(l))"
                for (i,l) in enumerate(LAMS)],"\n")

    # (5) Clamp schedule — also decided on calibration, and by a principle: never rectify per-voxel
    # before averaging (Jensen). The measurement only confirms what §6 already refuses.
    _clamp=[(s=sc, rmse=cal_pv_rmse(TV_LAMBDA;simplex=sc)) for sc in (:each,:once,:never)]

    # (6) HELD-OUT REPORTING ONLY — λ is already fixed. These re-score §8's own ROIs through the same
    # score_rois, sweeping λ, to SHOW the ROI-level consequence. Reading a λ off this table would be
    # exactly the leak this restructure removes, so the fitted λ is marked, not chosen, here.
    roi_stats(lam; simplex=TV_SIMPLEX) = (d=score_rois(lambda=lam, simplex=simplex);
        (m=[metrics([r.t[c] for r in d],[r.p[c] for r in d]) for c in 1:3], n=length(d)))
    LAMS2=sort(unique(vcat([0.0,3.0,10.0,30.0,100.0],TV_LAMBDA)))
    rs=[roi_stats(l) for l in LAMS2]
    i2_ship=findfirst(==(TV_LAMBDA),LAMS2)
    _rows2=join(["| $(l==0 ? "0 (raw)" : string(l)) | $(round(rs[i].m[1].ccc,digits=4)) | $(round(rs[i].m[2].ccc,digits=4)) | $(round(rs[i].m[3].ccc,digits=4)) | $(round(rs[i].m[2].slope,digits=3)) | $(round(rs[i].m[2].rmse,digits=4)) |$(l==TV_LAMBDA ? " ← **fitted λ**" : "")"
                for (i,l) in enumerate(LAMS2)],"\n")
    # CCC is nearly insensitive here — it is dominated by the huge between-ROI spread, so a real
    # degradation hides in its 3rd decimal. Judge the ROI cost on RMSE, and quote both.
    _dccc=rs[i2_ship].m[2].ccc-rs[1].m[2].ccc
    _rratio=rs[i2_ship].m[2].rmse/rs[1].m[2].rmse
    _roi_verdict = _rratio<1.05 ?
        "**ROI-level accuracy is unchanged** (f\\_l RMSE ×$(round(_rratio,digits=2)))." :
        "that is a **real ROI-level cost: f\\_l RMSE ×$(round(_rratio,digits=2))** ($(round(rs[i2_ship].m[2].rmse,digits=4)) vs $(round(rs[1].m[2].rmse,digits=4))), and it is stated rather than hidden behind CCC — CCC is dominated by the between-ROI spread and barely moves ($(round(_dccc,digits=4))) while RMSE rises $(round(Int,100*(_rratio-1)))%."

    Markdown.parse("""
**Noise measured** (calibration rods) — σ($(Int(EHI)) keV)=$(round(nz_hi.sd,digits=1)) HU, skew $(round(nz_hi.skew,digits=2)), excess kurtosis $(round(nz_hi.exkurt,digits=2)) ⇒ **Gaussian marginal**. Poisson is upstream, in the projections; each FBP voxel sums ~10³ rays, so the CLT leaves the Poisson origin visible only as the σ(HU) ladder, not as the shape. Lag-1 ACF = **$(round(acf_x[2],digits=2))** (x) / $(round(acf_y[2],digits=2)) (y) ⇒ **spatially correlated**, ≈$(round(acf_len,digits=1)) voxels per independent sample. Not modelled explicitly: that is a near-constant factor on w (a scan-geometry property, not a per-voxel one), so it rescales w uniformly and the fitted λ absorbs it whole.

**The σ\\_f weight was right; its scale was not.** median w=1/σ\\_f²=$(round(w_med,digits=1)), while `den = w + Σ_nbr λ/max(‖∇f‖,eps)`. At λ=0.05 (the original default) the 4 TV neighbours pulled **$(round(tv_pull(0.05),digits=3))×** the data — the TV was decorative and the "denoised" map was the raw decode. At the fitted λ=$(TV_LAMBDA) they pull $(round(tv_pull(TV_LAMBDA),digits=1))×.

## λ is fitted, on calibration

**λ = $(TV_LAMBDA)** — golden-section on log₁₀λ over [0.1, 100], $(_gold.n) evaluations, minimising **per-voxel RMSE vs GT across the $(length(CALPREP)) calibration thoraxes**. No grid, no literal: change the scanner, the dose or the keV pair and λ refits itself. Bracket check (λ/3, λ, λ×3): **$(_uni_ok ? "unimodal ✓" : "NOT unimodal ✗ — golden-section's assumption fails, treat λ as unverified")**.

The held-out scans (circular test + sector) are **not** in this objective and never were — they are reported in §8 and below, never optimised against.

| λ | calibration per-voxel RMSE |
|---|---|
$(_rows)

**Clamp schedule** (also calibration-only; `:each` = project every sweep, `:once` = project the result, `:never` = raw): per-voxel RMSE $(join(["`:$(c.s)` $(round(c.rmse,digits=4))" for c in _clamp], " · ")). The choice is principled before it is measured — rectifying each voxel before averaging is a Jensen bias on any region mean drawn from the map, which is the same thing §6 refuses for the pooled decode. `:$(TV_SIMPLEX)` ships.

## Held-out consequence (reported, not selected)

λ was fixed above. This table only shows what the fitted λ does to the region-mean statistics §8 reports — λ=0 is the raw decode fig5 used to report in its place:

| λ | f\\_w CCC | f\\_l CCC | f\\_p CCC | f\\_l slope | f\\_l RMSE |
|---|---|---|---|---|---|
$(_rows2)

At the fitted λ=$(TV_LAMBDA), $(_roi_verdict)

**This is the trade, and removing the leak is what exposed it.** The objective above is *per-voxel* RMSE — the map. It is not free at the region level, and it never was: averaging ~$(round(Int,mean(r.nvox for r in drois))) core voxels is already a ≈$(round(Int,sqrt(mean(r.nvox for r in drois))))× denoiser holding the insert boundary as an oracle the delivered map never gets, so TV has almost no variance left to remove there and mostly bias to add. A λ tuned to look good on *this* table would be tuned on held-out data — the exact leak §6 now forbids. So the honest options are: keep the per-voxel objective and accept the region-mean cost quoted above (what ships), or state an explicitly ROI-aware objective **and fit it on calibration too**. What is no longer available is reading a number off this table.

**The map is what TV is for; this table is what proves it did not cheat.**
""")
end

# ╔═╡ aaaa0032-0000-4000-8000-000000000032
# λ side-by-side: raw per-voxel decode vs the FITTED λ, on both geometries. Both panels run the real
# `deliver` path; only λ differs. λ comes from the calibration fit, so this figure follows the data.
# GT sits alongside because "which λ" is settled against truth, not by eye.
begin
    circ_raw = deliver(map_lo,map_hi,map_m2,mcomps; lambda=0.0)
    sect_raw = deliver(smap_lo,smap_hi,smap_m2,scomps; lambda=0.0)
    function pv_stats(rec,m2,comps,rpx)                      # per-voxel f_l vs GT on the eroded cores
        t=Float64[];r=Float64[];sds=Float64[]
        for k in 1:length(comps); ci=core_idx(m2,ROD0-1+k,rpx); isempty(ci)&&continue
            v=[rec[I,2] for I in ci if isfinite(rec[I,2])]; isempty(v)&&continue
            append!(t,fill(comps[k][2],length(v))); append!(r,v); push!(sds,std(v))
        end
        (rmse=sqrt(mean((r.-t).^2)), sd=mean(sds))
    end
    LROWS=((circ,circ_raw,map_m2,mcomps,CORE_RPX,"circular"),(sect,sect_raw,smap_m2,scomps,7,"sector"))
    let f=CM.Figure(size=(1180,780))
        for (row,(D,R,m2,comps,rpx,nm)) in enumerate(LROWS)
            hi=findall(!isnan,D.tru[:,:,2]); ci=extrema(getindex.(hi,1)); cj=extrema(getindex.(hi,2)); pad=18
            rI=max(1,ci[1]-pad):min(size(m2,1),ci[2]+pad); rJ=max(1,cj[1]-pad):min(size(m2,2),cj[2]+pad)
            sD=pv_stats(D.rec,m2,comps,rpx); sR=pv_stats(R.rec,m2,comps,rpx)
            _st(st)= st===nothing ? "" : @sprintf("\nper-voxel RMSE %.4f · within-core sd %.4f",st.rmse,st.sd)
            for (col,(img,ttl,st)) in enumerate(((D.tru[rI,rJ,2],"$nm · true f_l",nothing),
                                                 (R.rec[rI,rJ,2],"λ=0 — raw per-voxel decode",sR),
                                                 (D.rec[rI,rJ,2],"λ=$(TV_LAMBDA) — fitted, ships",sD)))
                ax=CM.Axis(f[row,col];title=ttl*_st(st),titlesize=11,aspect=CM.DataAspect(),yreversed=true)
                CM.hidedecorations!(ax); hm=CM.heatmap!(ax,img;colormap=:jet,colorrange=(0,1))
                (row==1&&col==3) && CM.Colorbar(f[:,4],hm;label="f_l")
            end
        end
        CM.Label(f[0,:],"f_l delivered map — raw decode vs fitted λ=$(TV_LAMBDA) (σ_f Huber-TV, simplex=:$(TV_SIMPLEX)); λ fitted on CALIBRATION by golden-section, held-out scans untouched\nROI-level check on the held-out n=$(length(drois)): f_l CCC $(round(rs[i2_ship].m[2].ccc,digits=4)) at λ=$(TV_LAMBDA) vs $(round(rs[1].m[2].ccc,digits=4)) raw — the map gains without the region means paying";fontsize=12,font=:bold)
        safe_save(joinpath(ASSET,"fig10_lambda_compare.png"),f); f
    end
end

# ╔═╡ aaaa0026-0000-4000-8000-000000000026
# Portable model snapshot — dumps the fitted surface + noise + gate + calibration table to
# wlp_model_<pair>.toml (stdlib TOML, no BasisSimulator), so the model can be applied outside
# this notebook. In-notebook, the `wlp_apply` cell below consumes the same coefficients live.
# The cal table (ROI means/stds + true fractions) lets a new chain refit the same recipe.
begin
    import TOML
    const MODEL_TOML = joinpath(@__DIR__, "wlp_model_$(PTAG).toml")
    open(MODEL_TOML, "w") do io
        TOML.print(io, Dict(
            "pair" => Dict("E_low_keV"=>ELO, "E_high_keV"=>EHI),
            "endpoints_hu" => Dict("water"=>collect(PW), "lipid"=>collect(PL), "protein"=>collect(PP)),
            "poly2" => Dict("basis"=>"[1, hLow, hHigh, hLow^2, hHigh^2, hLow*hHigh] -> (fw,fl,fp), normalized by sum",
                "cw"=>cw, "cl"=>cl, "cp"=>cp, "cl_affine"=>cl_aff),
            "noise" => Dict("sigma_quad_low"=>sc_lo, "sigma_quad_high"=>sc_hi, "rho"=>ρ,
                "Sigma_hu"=>[collect(Σhu[i,:]) for i in 1:2], "bias_hu"=>bias_hu),
            "gate" => Dict("soft_hu_lo"=>SOFT_HU_LO, "soft_hu_hi"=>SOFT_HU_HI),
            # TV belongs in the snapshot: wlp_apply denoises by default, so a consumer without these
            # reproduces a different map. λ is weighed against w=1/σ_f², hence O(10), not O(0.01).
            "tv" => Dict("lambda"=>TV_LAMBDA, "iters"=>TV_ITERS, "eps"=>TV_EPS, "simplex"=>String(TV_SIMPLEX),
                "form"=>"coupled Huber-TV on (f_l,f_p); den = w + Σ_nbr λ/max(‖∇f‖,eps), w = 1/σ_f²",
                "simplex_note"=>"project onto {f≥0, f_l+f_p≤1} ONCE on the result, never per sweep: per-sweep rectification is a Jensen bias on any region mean drawn from the map",
                "lambda_selected_by"=>"golden-section on log10(lambda) in [0.1,100], minimising per-voxel RMSE vs GT over the $(length(CALPREP)) CALIBRATION thoraxes only; held-out circular/sector scans never enter the objective",
                "lambda_transfers"=>"NO. This lambda is supervised by phantom GT. On real CT with no GT, refit it GT-free at the same noise level (discrepancy principle against the sigma ladder, GCV, or SURE) — do not copy this number across scanners/dose"),
            "provenance" => Dict("source"=>"wlp_decomposition.jl",
                "chain"=>"80/140kVp EICT (:dd_fast) -> Cong water/iodine -> FBP $(RECON_N)px/$(Int(RECON_FOV_MM))mm -> VMI $(PTAG) keV; stadium QRM-thorax",
                "cal_n"=>length(calrois), "r2_fw_fit"=>r2fit(cw,fwc), "test_n"=>length(drois),
                # scored through the delivered estimator (TV included) — same numbers as §8 and fig5
                "test_ccc"=>[mw.ccc, ml.ccc, mp.ccc], "test_rmse"=>[mw.rmse, ml.rmse, mp.rmse],
                "test_slope"=>[mw.slope, ml.slope, mp.slope]),
            "calibration_table" => Dict("fw"=>fwc, "fl"=>flc, "fp"=>fpc,
                "hu_low_mean"=>m_lo_cal, "hu_high_mean"=>m_hi_cal,
                "hu_low_std"=>[r.s_lo for r in calrois], "hu_high_std"=>[r.s_hi for r in calrois],
                "n_vox"=>[length(r.v_lo) for r in calrois]),
        ))
    end
    Markdown.parse("**Model snapshot** → `$(basename(MODEL_TOML))` (poly2 surface + noise + gate + calibration table; applied live by the `wlp_apply` cell below).")
end

# ╔═╡ aaaa0027-0000-4000-8000-000000000027
# One-click product: apply the fitted model to ANY co-registered VMI pair from the same 70/150-keV
# chain → (f_w, f_l, f_p) maps. No external file, no re-fit — the model IS the live fitted globals
# (cw,cl,cp,cl_aff, sc_lo,sc_hi, ρ, the soft-tissue gate); this reuses the delivered-map pipeline
# (fullfield · sigma_f_weight · tv_coupled) verbatim, so the product and the validation share one method.
begin
    # tv=true → boundary-agnostic delivered map (2D slice); tv=false → raw per-voxel decode (any dim).
    function wlp_apply(vmi_low, vmi_high; tv=true)
        fw, fl, fp = fullfield(vmi_low, vmi_high)
        tv || return (fw, fl, fp)
        gate = .!isnan.(fl); w = sigma_f_weight(vmi_low, vmi_high)
        fl_tv, fp_tv = tv_coupled(fl, fp, gate; lambda=TV_LAMBDA, w=w)
        fw_tv = map((a, b) -> isnan(a) ? NaN : 1 - a - b, fl_tv, fp_tv)
        (fw_tv, fl_tv, fp_tv)
    end
    let  # invariant: on the cached delivered-map slice, wlp_apply must reproduce the notebook's delivered map
        fw, fl, fp = wlp_apply(map_lo, map_hi)
        @assert isequal(cat(fw, fl, fp; dims=3), recmap) "wlp_apply must reproduce the delivered map"
        ng = count(!isnan, fl); lo, hi = extrema(filter(!isnan, fl))
        Markdown.parse("**One-click apply** — `fw, fl, fp = wlp_apply(vmi_low, vmi_high)` on a co-registered $(Int(ELO))/$(Int(EHI)) keV VMI pair. Reproduces the delivered map bit-for-bit on the cached slice: $ng gated voxels, f_l ∈ [$(round(lo,digits=2)), $(round(hi,digits=2))].")
    end
end

# ╔═╡ aaaa0028-0000-4000-8000-000000000028
# Raw export: dump every CT scan as an ImageJ-openable .raw (full 512×512×3 z-stack), grouped by role
# into data/recon/{calibration,test}/, with per-label truth in label_comps.csv. Reuses the LIVE sim
# arrays (no .jls re-read). Column-major Float32/UInt8, little-endian, NOT dim2-reversed (figures use
# yreversed=true ⇒ ImageJ row 0 = top). Circular sims share the calibration label stack.
let
    RECON = joinpath(@__DIR__, "data", "recon")
    dims3(M) = "$(size(M,1))x$(size(M,2))x$(size(M,3))"
    wf32(dir, base, kev, M) = write(joinpath(dir, "$(base)_vmi$(kev)keV_$(dims3(M))_float32.raw"), Array{Float32}(M))
    wu8(dir, base, M)       = write(joinpath(dir, "$(base)_$(dims3(M))_uint8.raw"), Array{UInt8}(M))
    scan(dir, base, h_lo, h_hi) = (wf32(dir, base, Int(ELO), h_lo); wf32(dir, base, Int(EHI), h_hi))
    comps_csv(dir, rows) = open(joinpath(dir, "label_comps.csv"), "w") do io
        println(io, "scan,label,f_water,f_lipid,f_protein")
        for (name, comps) in rows, (k, c) in enumerate(comps)
            @printf(io, "%s,%d,%.4f,%.4f,%.4f\n", name, ROD0 - 1 + k, c[1], c[2], c[3])
        end
    end
    rm(RECON; recursive=true, force=true)
    CAL = joinpath(RECON, "calibration"); TST = joinpath(RECON, "test"); mkpath(CAL); mkpath(TST)
    NZ = size(map_hi_v, 3)
    crows = Tuple{String,Any}[]
    for (i, s) in enumerate(calsims); scan(CAL, "noise_sim$i", s.hu_lo, s.hu_hi); push!(crows, ("noise_sim$i", s.comps)); end
    scan(CAL, "deliveredmap", map_lo_v, map_hi_v); push!(crows, ("deliveredmap", mcomps))
    wu8(CAL, "labels", repeat(map_m2, 1, 1, NZ)); comps_csv(CAL, crows)
    trows = Tuple{String,Any}[]
    for (i, s) in enumerate(testsims); scan(TST, "circular_sim$i", s.hu_lo, s.hu_hi); push!(trows, ("circular_sim$i", s.comps)); end
    for (i, s) in enumerate(sectsims); scan(TST, "sector_sim$i", s.hu_lo, s.hu_hi); push!(trows, ("sector_sim$i", s.comps)); end
    scan(TST, "sector_deliveredmap", smap_lo_v, smap_hi_v); push!(trows, ("sector_deliveredmap", scomps))
    wu8(TST, "labels_circular", repeat(map_m2, 1, 1, NZ)); wu8(TST, "labels_sector", repeat(smap_m2, 1, 1, NZ)); comps_csv(TST, trows)
    p = joinpath(CAL, "noise_sim1_vmi$(Int(ELO))keV_512x512x$(NZ)_float32.raw")   # byte-layout round-trip
    @assert reshape(reinterpret(Float32, read(p)), 512, 512, NZ) == Array{Float32}(calsims[1].hu_lo) "raw round-trip mismatch — byte layout wrong"
    write(joinpath(RECON, "README_imagej.txt"), """
ImageJ → File → Import → Raw…
  Image type       = 32-bit Real  (*_float32.raw)   |   8-bit  (*_labels*_uint8.raw)
  Width = nx, Height = ny, Number of images = nz (= $(NZ))   ·   ☑ Little-endian byte order
Column-major, NOT dim2-reversed (notebook figures use yreversed=true ⇒ ImageJ row 0 = top).
calibration/: noise_sim1..$(length(calsims)) + deliveredmap ; labels = shared insert stack (8..20).
test/:  circular_sim1..$(length(testsims)) · sector_sim1..$(length(sectsims)) + sector_deliveredmap
        labels_circular (8..20) · labels_sector (8..23).
VMI keV pair: $(Int(ELO)) / $(Int(EHI)).  label_comps.csv (per folder) → each scan's per-label
(f_water, f_lipid, f_protein) ground truth.
""")
    Markdown.parse("**Raws exported** → `data/recon/` — calibration: $(2*(length(calsims)+1)) scans, test: $(2*(length(testsims)+length(sectsims)+1)) scans (ImageJ 32-bit, little-endian, $(NZ)-slice stacks). Layout in `README_imagej.txt`.")
end

# ╔═╡ aaaa0015-0000-4000-8000-000000000015
md"## 7 · Figures"

# ╔═╡ aaaa0016-0000-4000-8000-000000000016
# Barycentric triangle + noise-ellipse MLE, in ONE figure: Panel A = the full W/L/P sliver
# (every test voxel, coloured by whether its raw decode is feasible); Panel B = a zoom on the
# water–lipid corner showing the Σ noise ellipse and the out-of-triangle ROI means projected
# back onto the triangle in the Mahalanobis (Σ⁻¹) metric. Rebuilds from the decode-cell globals.
let
    # ── per-voxel HU + feasibility across all scored ROIs (circular + sector) ──
    HUx=Float64[]; HUy=Float64[]; feas=Bool[]
    for r in allrois, j in eachindex(r.v_lo)
        d=decode(r.v_lo[j],r.v_hi[j]); push!(HUx,r.v_lo[j]); push!(HUy,r.v_hi[j]); push!(feas, all(d.≥-1e-9))
    end
    stride=max(1,length(HUx)÷5000); ss=1:stride:length(HUx)                       # deterministic thin for plotting
    fin=[i for i in ss if feas[i]]; fout=[i for i in ss if !feas[i]]
    pv_out=100*count(!,feas)/length(feas)
    rmx=[r.m_lo for r in allrois]; rmy=[r.m_hi for r in allrois]
    rout=findall(any(decode(r.m_lo,r.m_hi).<-1e-6) for r in allrois)                 # ROI means outside the simplex
    condG=cond(Gmat); s70v=sqrt(Σhu[1,1]); s150v=sqrt(Σhu[2,2])
    CIN=CM.RGBf(0.353,0.655,0.353); COUT=CM.RGBf(0.557,0.373,0.659)                # feasible / infeasible
    CWv=CM.RGBf(0.231,0.459,0.690); CLv=CM.RGBf(0.910,0.639,0.239); CPv=CM.RGBf(0.757,0.267,0.235)
    zx0,zx1,zy0,zy1=-125.0,20.0,-95.0,15.0                                          # zoom window (water–lipid corner)

    f=CM.Figure(size=(1180,520))
    # ── Panel A: the whole triangle (equal aspect ⇒ the sliver is honest) ──
    axA=CM.Axis(f[1,1];xlabel="HU @ $(Int(ELO)) keV",ylabel="HU @ $(Int(EHI)) keV",
                title="The W/L/P triangle is a sliver (cond G = $(round(condG,digits=1)))",aspect=CM.DataAspect())
    CM.poly!(axA,[CM.Point2f(PW...),CM.Point2f(PL...),CM.Point2f(PP...)];color=(:gray,0.10),strokecolor=(:black,0.45),strokewidth=1.2)
    CM.scatter!(axA,HUx[fin],HUy[fin];color=(CIN,0.30),markersize=3,label="decode inside")
    CM.scatter!(axA,HUx[fout],HUy[fout];color=(COUT,0.35),markersize=3,label="decode outside ($(round(pv_out,digits=1))%)")
    for (p,c,t) in ((PW,CWv,"water"),(PL,CLv,"lipid"),(PP,CPv,"protein"))
        CM.scatter!(axA,[p[1]],[p[2]];color=c,markersize=13,strokecolor=:white,strokewidth=1.2)
        CM.text!(axA,p[1],p[2];text=t,fontsize=11,color=c,font=:bold,align=(:left,:bottom))
    end
    CM.lines!(axA,[zx0,zx1,zx1,zx0,zx0],[zy0,zy0,zy1,zy1,zy0];color=(:black,0.6),linestyle=:dash,linewidth=1.0)
    CM.text!(axA,zx1,zy1;text="B",fontsize=11,font=:bold,color=(:black,0.7),align=(:left,:bottom))
    CM.axislegend(axA;position=:lt,framevisible=false,labelsize=9)
    # ── Panel B: zoom on the water–lipid corner + noise ellipse + MLE projections ──
    axB=CM.Axis(f[1,2];xlabel="HU @ $(Int(ELO)) keV",ylabel="HU @ $(Int(EHI)) keV",
                title="Out-of-triangle ROI means → Σ⁻¹ (Mahalanobis) MLE projection",limits=(zx0,zx1,zy0,10.0))
    CM.poly!(axB,[CM.Point2f(PW...),CM.Point2f(PL...),CM.Point2f(PP...)];color=(:gray,0.10),strokecolor=(:black,0.45),strokewidth=1.2)
    inz=[i for i in 1:length(HUx) if zx0≤HUx[i]≤zx1 && zy0≤HUy[i]≤zy1]
    inz=inz[1:max(1,length(inz)÷4000):end]
    CM.scatter!(axB,HUx[[i for i in inz if feas[i]]],HUy[[i for i in inz if feas[i]]];color=(CIN,0.22),markersize=3)
    CM.scatter!(axB,HUx[[i for i in inz if !feas[i]]],HUy[[i for i in inz if !feas[i]]];color=(COUT,0.26),markersize=3)
    ev=eigen(Σhu); anchor=[-84.0,-55.0]                                            # ellipse in empty space below the cloud
    for k in (1.0,2.0)
        pts=[anchor.+k.*(sqrt(ev.values[1])*cos(τ).*ev.vectors[:,1].+sqrt(ev.values[2])*sin(τ).*ev.vectors[:,2]) for τ in range(0,2π,length=140)]
        CM.lines!(axB,first.(pts),last.(pts);color=:black,linewidth=(k==1.0 ? 1.6 : 1.0),linestyle=(k==1.0 ? :solid : :dash))
    end
    CM.scatter!(axB,[anchor[1]],[anchor[2]];color=:black,markersize=5)
    CM.text!(axB,anchor[1],anchor[2]-7;text="noise Σ  (σ₇₀=$(round(s70v,digits=1)), σ₁₅₀=$(round(s150v,digits=1)), ρ=$(round(ρ,digits=2)))\n1σ / 2σ contours",fontsize=8,align=(:center,:top))
    for i in rout                                                                  # measured mean → feasible edge point
        θ=Gmat\([rmx[i],rmy[i]].-bias_hu); (fl,fp)=proj_simplex((θ[1],θ[2])); hp=Gmat*[fl,fp]
        CM.lines!(axB,[rmx[i],hp[1]],[rmy[i],hp[2]];color=COUT,linewidth=1.3)
        CM.scatter!(axB,[hp[1]],[hp[2]];marker=:utriangle,color=COUT,markersize=8)
    end
    CM.scatter!(axB,rmx[rout],rmy[rout];color=COUT,markersize=10,strokecolor=:white,strokewidth=0.9)
    CM.scatter!(axB,[rmx[i] for i in 1:length(rmx) if !(i in rout)],[rmy[i] for i in 1:length(rmy) if !(i in rout)];color=(CIN,0.6),markersize=7)
    for (p,c,t) in ((PW,CWv,"water"),(PL,CLv,"lipid")); CM.scatter!(axB,[p[1]],[p[2]];color=c,markersize=13,strokecolor=:white,strokewidth=1.2); CM.text!(axB,p[1],p[2];text=t,fontsize=11,color=c,font=:bold,align=(:left,:bottom)); end
    CM.Label(f[0,:],"Decomposition triangle + noise model: $(round(pv_out,digits=1))% of voxels decode outside the simplex under noise (96% within 2σ); only $(length(rout))/$(length(allrois)) ROI means need the pooled MLE projection";fontsize=12,font=:bold)
    safe_save(joinpath(ASSET,"fig_decode_triangle_noise.png"),f); f
end

# ╔═╡ aaaa0017-0000-4000-8000-000000000017
let f=CM.Figure(size=(1050,460))
    ax=CM.Axis(f[1,1];xlabel="HU",ylabel="σ (HU)",title="σ(HU) per energy — convex-quadratic")
    for (hu,sg,cc,e,col) in ((m_lo_cal,[r.s_lo for r in calrois],sc_lo,ELO,:tomato),(m_hi_cal,[r.s_hi for r in calrois],sc_hi,EHI,:royalblue))
        CM.scatter!(ax,hu,sg;color=col,markersize=8,label="$(Int(e)) keV"); g=range(minimum(hu),maximum(hu),100); CM.lines!(ax,g,quad_sigma.(Ref(cc),g);color=col)
    end
    CM.axislegend(ax;position=:rt)
    ax2=CM.Axis(f[1,2];xlabel="resid HU$(Int(ELO))",ylabel="resid HU$(Int(EHI))",title="inter-energy ρ=$(round(ρ,digits=2))")
    idx=rand(1:length(res_lo),min(3000,length(res_lo))); CM.scatter!(ax2,res_lo[idx],res_hi[idx];markersize=3,color=(:purple,0.3))
    safe_save(joinpath(ASSET,"fig2_noise.png"),f); f
end

# ╔═╡ aaaa0018-0000-4000-8000-000000000018
let f=CM.Figure(size=(1520,430))
    ax=CM.Axis(f[1,1];title="VMI $(Int(EHI)) keV",aspect=CM.DataAspect(),yreversed=true); CM.hidedecorations!(ax); CM.heatmap!(ax,map_hi;colormap=:grays,colorrange=(-200,300))
    for (col,(img,ttl)) in enumerate(((recmap[:,:,1],"f_w"),(recmap[:,:,2],"f_l"),(recmap[:,:,3],"f_p")))
        ax2=CM.Axis(f[1,col+1];title=ttl,aspect=CM.DataAspect(),yreversed=true); CM.hidedecorations!(ax2); CM.heatmap!(ax2,img;colormap=:jet,colorrange=(0,1))
    end
    CM.Colorbar(f[1,5];colormap=:jet,colorrange=(0,1),label="volume fraction")
    CM.Label(f[0,:],"Delivered map — per-voxel decode + σ_f-weighted Huber-TV (λ=$(TV_LAMBDA), simplex=:$(TV_SIMPLEX)), boundary-agnostic (lung & bone HU-gated out)";fontsize=13,font=:bold)
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

# ╔═╡ aaaa0022-0000-4000-8000-000000000022
let f=CM.Figure(size=(1520,430))
    ax=CM.Axis(f[1,1];title="VMI $(Int(EHI)) keV (sector)",aspect=CM.DataAspect(),yreversed=true); CM.hidedecorations!(ax); CM.heatmap!(ax,smap_hi;colormap=:grays,colorrange=(-200,300))
    for (col,(img,ttl)) in enumerate(((sect.rec[:,:,1],"f_w"),(sect.rec[:,:,2],"f_l"),(sect.rec[:,:,3],"f_p")))
        ax2=CM.Axis(f[1,col+1];title=ttl,aspect=CM.DataAspect(),yreversed=true); CM.hidedecorations!(ax2); CM.heatmap!(ax2,img;colormap=:jet,colorrange=(0,1))
    end
    CM.Colorbar(f[1,5];colormap=:jet,colorrange=(0,1),label="volume fraction")
    CM.Label(f[0,:],"Delivered map — sector validation phantom (per-voxel decode + σ_f Huber-TV λ=$(TV_LAMBDA), simplex=:$(TV_SIMPLEX), boundary-agnostic)";fontsize=13,font=:bold)
    safe_save(joinpath(ASSET,"fig7_delivered_map_sector.png"),f); f
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
    CM.Label(f[0,:],"Recovered vs true (n=$(length(drois)): $(count(==(:circular),geomtag)) circular + $(count(==(:sector),geomtag)) sector · DELIVERED estimator: per-voxel decode + σ_f Huber-TV λ=$(TV_LAMBDA), simplex=:$(TV_SIMPLEX) · eroded-core pool) — blue=circular, orange=sector; error bars = SE of the ROI mean";fontsize=12,font=:bold)
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

# ╔═╡ aaaa0025-0000-4000-8000-000000000025
Markdown.parse("""
## 8 · Conclusion

| fraction | CCC | slope | RMSE |
|---|---|---|---|
| **f_water** | **$(round(mw.ccc,digits=3))** | $(round(mw.slope,digits=2)) | $(round(mw.rmse,digits=3)) |
| f_lipid | $(round(ml.ccc,digits=3)) | $(round(ml.slope,digits=2)) | $(round(ml.rmse,digits=3)) |
| f_protein | $(round(mp.ccc,digits=3)) | $(round(mp.slope,digits=2)) | $(round(mp.rmse,digits=3)) |

Held-out **circular + sector** (n=$(length(drois))), scored through the **delivered** estimator — per-voxel
decode + σ\\_f-weighted Huber-TV (λ=$(TV_LAMBDA), simplex=`:$(TV_SIMPLEX)`), the same chain `wlp_apply` ships, so this table and the
delivered map (fig 3/7) cannot disagree. Detectability: **$(round(Int,100mean(dHU_hi.<5)))% of ROIs < 5 HU at $(Int(EHI)) keV** (mean $(round(mean(dHU_hi),digits=1)) HU).

**keV pair — $(Int(ELO))/$(Int(EHI)).** This pair is intentionally ill-conditioned (150 keV is a clinically
standard VMI, but the two energies sit above the photoelectric-rich low-keV regime): the W/L/P triangle is a
near-collinear sliver, **cond(G) = $(round(cond([PL[1]-PW[1] PP[1]-PW[1]; PL[2]-PW[2] PP[2]-PW[2]]),digits=1))**
(≈3.5× a low-keV pair). Yet the held-out **ROI CCC is unaffected** — the √N eroded-core pooling absorbs the
per-voxel conditioning penalty, and the inter-energy noise correlation ρ=$(round(ρ,digits=2)) is low enough to
help the separation. The penalty surfaces only in the per-voxel maps and the integrated-HU total; conditioning
number alone overpredicts it. The pair is a single `WLP_PAIR` knob.

**Point accuracy** is excellent on the eroded interior cores (all CCC ≥ $(round(minimum((mw.ccc,ml.ccc,mp.ccc)),digits=3))) and, as expected on a uniform
phantom, per-voxel vs pool-then-decode barely differ there (f\\_l CCC $(round(ml.ccc,digits=3)) vs $(round(ml_pool.ccc,digits=3))). The honesty cost of the ground-truth
boundary shows up in the **delivered map** (fig 3–4): the boundary-agnostic per-voxel+TV map keeps real texture
and PVE edges, whereas the GT-pooled map is flat because it uses a boundary real fat doesn't provide.

**The denoiser is fitted, not admired.** λ=$(TV_LAMBDA) is not a setting: it is the golden-section minimiser of
per-voxel RMSE vs GT over the **calibration** thoraxes ($(_gold.n) evaluations on log₁₀λ ∈ [0.1,100]; §6.5). The
held-out circular and sector scans are absent from that objective, so the table above is a test score, not a
training score — and λ carries no literal, so a new scanner, dose or keV pair refits it.

**What it costs, plainly.** The objective is the *map*, and the map is not free at the region level: vs the raw
decode, the fitted λ leaves f\\_l CCC at $(round(ml.ccc,digits=4)) but multiplies ROI f\\_l RMSE by
**$(round(_rratio,digits=2))×**. That is the honest price of a per-voxel objective, and it is quoted rather than
buried in CCC's third decimal — CCC is dominated by the between-ROI spread and is nearly blind to this. Anyone
who needs region means more than maps should state an ROI-aware objective **and fit it on calibration**; what is
not allowed is tuning λ against the held-out table.

Three traps are worth naming. Smoothness is not accuracy: within-core σ falls monotonically with λ while RMSE
turns back up, so an eye-tuned λ over-smooths and pays in bias. The simplex projection must fire **once**, not
per sweep — rectifying every sweep is the same per-voxel Jensen bias this notebook refuses for the pooled decode
(and on calibration `:once` also happens to beat both `:each` and `:never` on per-voxel RMSE). And λ here is
supervised by phantom GT, so it **does not transfer to real CT**: with no GT, refit it GT-free at the measured
noise level — the discrepancy principle against the σ ladder, GCV, or SURE — rather than copying this number.

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
# ╟─aaaa0008-0000-4000-8000-000000000008
# ╠═aaaa0029-0000-4000-8000-000000000029
# ╟─aaaa0009-0000-4000-8000-000000000009
# ╠═aaaa0010-0000-4000-8000-000000000010
# ╠═aaaa0030-0000-4000-8000-000000000030
# ╟─aaaa0011-0000-4000-8000-000000000011
# ╠═aaaa0012-0000-4000-8000-000000000012
# ╟─aaaa0013-0000-4000-8000-000000000013
# ╠═aaaa0014-0000-4000-8000-000000000014
# ╠═aaaa0031-0000-4000-8000-000000000031
# ╠═aaaa0032-0000-4000-8000-000000000032
# ╠═aaaa0026-0000-4000-8000-000000000026
# ╠═aaaa0027-0000-4000-8000-000000000027
# ╠═aaaa0028-0000-4000-8000-000000000028
# ╟─aaaa0015-0000-4000-8000-000000000015
# ╟─aaaa0016-0000-4000-8000-000000000016
# ╟─aaaa0017-0000-4000-8000-000000000017
# ╟─aaaa0018-0000-4000-8000-000000000018
# ╟─aaaa0019-0000-4000-8000-000000000019
# ╟─aaaa0023-0000-4000-8000-000000000023
# ╟─aaaa0022-0000-4000-8000-000000000022
# ╟─aaaa0024-0000-4000-8000-000000000024
# ╟─aaaa0020-0000-4000-8000-000000000020
# ╟─aaaa0021-0000-4000-8000-000000000021
# ╟─aaaa0025-0000-4000-8000-000000000025
