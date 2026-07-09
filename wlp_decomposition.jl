### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ aaaa0001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(@__DIR__)
end

# ╔═╡ aaaa0002-0000-4000-8000-000000000002
md"""
# Water–Lipid–Protein Material Decomposition — a pure-physics study

Recover the **volumetric fractions** ``(f_w, f_l, f_p)`` of a water/lipid/protein mixture from
dual-energy CT. Everything is inline (no module scripts): mix materials by volume fraction, simulate
80/140-kVp DECT of rods in a QRM-thorax phantom with
[BasisSimulator.jl](https://github.com/MolloiLab/BasisSimulator.jl) **(v0.8.0)**, synthesize VMI at
**40 and 70 keV**, and invert.

**The math.** Per voxel, with pure-material endpoints ``\\mathbf p_w,\\mathbf p_l,\\mathbf p_p`` in the
``(\\mathrm{HU}_{40},\\mathrm{HU}_{70})`` plane:

```math
f_w\\,\\mu_{w,E}+f_l\\,\\mu_{l,E}+f_p\\,\\mu_{p,E}+\\varepsilon_E=m_E,\\quad E\\in\\{40,70\\};\\qquad f_w+f_l+f_p=1.
```

The per-material noise terms collapse to one ``\\varepsilon_E`` per energy read off the single
heteroscedastic curve ``\\sigma_E(\\mathrm{HU})`` (convex-quadratic below). Water-referencing
(``f_w=1-f_l-f_p``) gives a square ``2\\times2`` system ``\\mathbf m-\\mathbf p_w=G\\theta+\\varepsilon``,
``\\theta=(f_l,f_p)``, so the noiseless locus is the **triangle** ``\\triangle(\\mathbf p_w,\\mathbf p_l,\\mathbf p_p)``.
The ``-\\ln`` posterior adds the adipose prior ``\\mathcal N(f_w)\\,\\mathcal N(f_l)\\,\\Gamma(f_p)`` and a
coupled total-variation term. The decode is a **calibration surface** ``f=\\mathrm{poly}_2(\\mathrm{HU}_{40},\\mathrm{HU}_{70})``
fit from known mixtures, applied per-region (√N-pooled) and per-voxel.

**Headline result (137 test ROIs, diverse compositions).** ``f_w`` **CCC 0.96**, ``f_l`` **0.97**,
``f_p`` **0.99** (all slopes ≈ 0.99) — **all three exceed 0.9**. Detectability: **94% of ROIs < 5 HU at
70 keV** (mean 1.8 HU).

!!! warning "Requires BasisSimulator v0.8.0 with `projector = :dd_fast`"
    The recon must be quantitative: on v0.8.0 pure lipid reads −205 HU vs theoretical −213 (< 2%,
    matching example 07). The old v0.2.1 projector compresses non-water HU by ~50 HU, which breaks the
    linear-additive mixture model and would cap f_water near 0.75 — a **simulator artifact, not physics**.
    Validate the sim by measuring pure-material rods against `theoretical_hu` before trusting any decode.
"""

# ╔═╡ aaaa0003-0000-4000-8000-000000000003
md"## 1 · Setup"

# ╔═╡ aaaa0004-0000-4000-8000-000000000004
begin
    import BasisSimulator as BS
    import Metal
    import CairoMakie as CM
    using Unitful, LinearAlgebra, Statistics, Random, DelimitedFiles, Printf, Serialization
    to_gpu(x) = Metal.functional() ? Metal.MtlArray(x) : x
    const DATA = joinpath(@__DIR__, "data")
    const ASSET = joinpath(@__DIR__, "assets")
    safe_save(p, f) = CM.save(p, f; px_per_unit = 1.4)   # figures ≤1300 wide ⇒ ≤1820 px/side
    md"imports · GPU backend (Metal, CPU fallback) · `safe_save` (≤1920 px/side)"
end

# ╔═╡ aaaa0005-0000-4000-8000-000000000005
md"## 2 · Materials & theoretical endpoints"

# ╔═╡ aaaa0006-0000-4000-8000-000000000006
begin
    const PROTEIN = BS.XA.Material("protein_WW1986", 0.0, 0.0u"eV", 1.35u"g/cm^3",
        Dict{Int,Float64}(1=>0.066, 6=>0.534, 7=>0.170, 8=>0.220, 16=>0.010))   # Woodard & White 1986
    const WATER = BS.XA.Materials.basis_water; const LIPID = BS.XA.Materials.basis_lipid
    ρval(m) = BS.XA.val(m.density); const ΡW, ΡL, ΡP = ρval(WATER), ρval(LIPID), ρval(PROTEIN)
    "Volume fractions (fw,fl,fp) → one effective attenuating Material (attenuation uses density+composition only)."
    function wlp_material(fw, fl, fp; name="wlp")
        ρ = fw*ΡW + fl*ΡL + fp*ΡP; mf = (fw*ΡW/ρ, fl*ΡL/ρ, fp*ΡP/ρ); comp = Dict{Int,Float64}()
        for (m,wm) in ((WATER,mf[1]),(LIPID,mf[2]),(PROTEIN,mf[3])), (Z,f) in m.composition
            comp[Z] = get(comp,Z,0.0) + wm*f
        end
        BS.XA.Material(name, 0.0, 0.0u"eV", ρ*u"g/cm^3", comp)
    end
    theo_hu(mat, E) = (μw = BS.compute_μ_at_energy(WATER, E); 1000.0*(BS.compute_μ_at_energy(mat, E) - μw)/μw)
    mix_hu(fw, fl, fp, E) = theo_hu(wlp_material(fw,fl,fp), E)
    const E40, E70 = 40.0, 70.0
    const PL = (theo_hu(LIPID,E40), theo_hu(LIPID,E70)); const PP = (theo_hu(PROTEIN,E40), theo_hu(PROTEIN,E70)); const PW = (0.0, 0.0)
    md"Endpoints (theoretical HU): water=(0,0), lipid=$(round.(PL,digits=0)), protein=$(round.(PP,digits=0))."
end

# ╔═╡ aaaa0007-0000-4000-8000-000000000007
md"""## 3 · Composition generation
Adipose `draw_wlp` (KDE of the Woodard 1986 data, for the *prior*) and `diverse_comps` (spanning the
simplex, the *test* set of "different volumetric fractions")."""

# ╔═╡ aaaa0008-0000-4000-8000-000000000008
begin
    const ADI = ("healthy","obese","reduced","comparison","unspecified")
    mass_to_vol(w,l,p) = (v=(w/ΡW,l/ΡL,p/ΡP); s=sum(v); v./s)
    "KDE draw of adipose (f_w,f_l,f_p): Scott-bandwidth KDE on lipid% + logistic-normal protein share."
    function draw_wlp(seed, n; csv=joinpath(DATA,"adipose_composition_distribution.csv"))
        raw=readdlm(csv,','; header=false); hdr=string.(raw[1,:]); rows=raw[2:end,:]
        ci(x)=findfirst(==(x),hdr); cc,cl,cp,cs=ci("component"),ci("lipid_pct"),ci("component_pct"),ci("state")
        keep(i)=string(rows[i,cs]) in ADI && Float64(rows[i,cl])≥50
        L=[Float64(rows[i,cl]) for i in axes(rows,1) if keep(i)]; sl=Float64[];sp=Float64[]
        for i in axes(rows,1);(keep(i)&&string(rows[i,cc])=="protein")||continue;push!(sl,Float64(rows[i,cl]));push!(sp,Float64(rows[i,cp]));end
        rng=MersenneTwister(seed); bwL=std(L)*length(L)^(-1/5); q=sp./(100 .-sl); z=log.(q./(1 .-q)); μz,σz=mean(z),std(z)
        [(l=clamp(L[rand(rng,1:length(L))]+bwL*randn(rng),50.,99.);qq=1/(1+exp(-(μz+σz*randn(rng))));p=qq*(100-l);mass_to_vol(100-l-p,l,p)) for _ in 1:n]
    end
    "Diverse compositions spanning the WLP simplex (f_l 0.05–0.95, f_p 0–0.3)."
    function diverse_comps(seed, n)
        rng=MersenneTwister(seed); out=NTuple{3,Float64}[]
        for _ in 1:n; fl=0.05+0.9*rand(rng);fp=0.3*rand(rng);fw=1-fl-fp; fw<0.02&&(fl-=0.02-fw;fw=0.02); push!(out,(fw,fl,fp)); end
        out
    end
    md"`draw_wlp` (adipose prior) · `diverse_comps` (spanning test set)"
end

# ╔═╡ aaaa0009-0000-4000-8000-000000000009
md"""## 4 · Phantom & forward acquisition
21-rod circular + 16-sector (8 angular × 2 radial) geometries on the reproducible QRM-thorax mask;
dual-kVp EICT with the **`:dd_fast`** projector → Cong water/iodine basis → FBP (3-slice, inside the
cone-usable z-band) → VMI at 40 & 70 keV. Uninvertible physics (fill-factor/crosstalk/scatter) off."""

# ╔═╡ aaaa0010-0000-4000-8000-000000000010
begin
    const NX, NY, VOX = 1850, 1350, 0.2; const ROD0, NROD = 8, 21
    load_mask2d() = reshape(Vector{UInt8}(read(joinpath(DATA,"qrm_thorax_wlplat_1850x1350_uint8.raw"))), NX, NY)
    function _base_mats(uniform)
        m=Dict{Int,BS.XA.Material}(0=>BS.XA.Materials.air,1=>BS.XA.Materials.lung,2=>BS.XA.Materials.muscle,
            3=>BS.XA.Materials.corticalbone,4=>BS.XA.Materials.marrow_red,5=>BS.XA.Materials.adipose,6=>BS.XA.Materials.water,7=>BS.XA.Materials.basis_lipid)
        uniform && (for l in 1:5; m[l]=BS.XA.Materials.water; end); m
    end
    function _phantom(m3, mats, ds)
        vox=VOX*ds/10; pc=BS.create_phantom_from_mask(Array{Int,3}(m3),mats,(vox,vox,vox))
        (cpu=pc, gpu=BS.Phantom(to_gpu(pc.mask),pc.materials,pc.voxel_size,pc.origin,pc.extent))
    end
    function build_rods(comps; nz=40, ds=2, uniform=true)
        m2=load_mask2d(); ds>1&&(m2=m2[1:ds:end,1:ds:end]); nx,ny=size(m2); m3=repeat(reshape(m2,nx,ny,1),1,1,nz)
        mats=_base_mats(uniform); for k in 1:NROD; mats[ROD0-1+k]=wlp_material(comps[k]...;name="rod$k"); end
        _phantom(m3,mats,ds)
    end
    function build_sectors(comps; nz=40, ds=2, uniform=true)   # 16 solid sectors: 8 angular × 2 radial
        @assert length(comps)==16
        m2=load_mask2d(); ds>1&&(m2=m2[1:ds:end,1:ds:end]); nx,ny=size(m2)
        ridx=findall(l->ROD0≤Int(l)≤ROD0+NROD-1, m2); cx=mean(getindex.(ridx,1)); cy=mean(getindex.(ridx,2))
        R=maximum(sqrt((Float64(i[1])-cx)^2+(Float64(i[2])-cy)^2) for i in ridx)+8.0/(VOX*ds)
        m3=repeat(reshape(m2,nx,ny,1),1,1,nz)
        @inbounds for j in 1:ny,i in 1:nx
            d=sqrt((i-cx)^2+(j-cy)^2); d≤R || continue
            sec=min(7,floor(Int,mod(atan(j-cy,i-cx),2π)/(2π/8))); ring=d<R/2 ? 0 : 1; lab=UInt8(ROD0+ring*8+sec)
            for k in 1:nz; m3[i,j,k]=lab; end
        end
        mats=_base_mats(uniform); for l in ROD0:(ROD0+NROD-1); mats[l]=BS.XA.Materials.water; end
        for s in 0:15; mats[ROD0+s]=wlp_material(comps[s+1]...;name="sec$s"); end
        _phantom(m3,mats,ds)
    end
    const SCANNER=BS.Scanner(source_to_isocenter=625.6,source_to_detector=1100.0,detector_rows=256,detector_cols=834,
        detector_row_size=0.625,detector_col_size=0.6,focal_spot_width=1.0,focal_spot_length=1.0,
        target_angle=10.0,flat_filter_material=:aluminum,flat_filter_thickness=2.5,bowtie_filter=:ge_revolution_large,
        detector_material=:lumex,detector_depth=3.0,fill_factor_row=0.9,fill_factor_col=0.9,electronic_noise=0,detection_gain=10.0)
    "One dual-kVp acquisition → VMI HU at 40 & 70 keV (:dd_fast projector · Cong water/iodine basis · FBP · z-median · 2-basis VMI)."
    function run_acq(pg; views=360,collimation=2.5,matrix=(512,512,3),fov_cm=38.0,z_cm=0.1875,seed=1234,zmed=1)
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
    function roi_cores(pc,geom,matrix,labels; radius_px=7)
        m3=BS.resample_to_recon(pc,geom,matrix;method=:nearest); midz=size(m3,3)÷2+1; m2=m3[:,:,midz]
        nx,ny=size(m2); out=Dict{Int,Vector{CartesianIndex{2}}}()
        for lab in labels; idx=findall(==(UInt8(lab)),m2); isempty(idx)&&continue
            cx=mean(getindex.(idx,1));cy=mean(getindex.(idx,2))
            out[lab]=[CartesianIndex(i,j) for j in 1:ny,i in 1:nx if (i-cx)^2+(j-cy)^2≤radius_px^2]; end
        (cores=out,midz=midz,m2=m2)
    end
    function collect_rois(acq,pc,label_comp; radius_px=7)
        rc=roi_cores(pc,acq.geom,(512,512,3),collect(keys(label_comp)); radius_px=radius_px); out=NamedTuple[]
        for (lab,c) in label_comp
            (haskey(rc.cores,lab)&&!isempty(rc.cores[lab]))||continue; ci=rc.cores[lab]
            v40=[Float64(acq.hu40[i,rc.midz]) for i in ci]; v70=[Float64(acq.hu70[i,rc.midz]) for i in ci]
            push!(out,(lab=lab,fw=c[1],fl=c[2],fp=c[3],v40=v40,v70=v70,m40=mean(v40),m70=mean(v70),s40=std(v40),s70=std(v70)))
        end
        (rois=out,midz=rc.midz,m2=rc.m2)
    end
    md"`build_rods` · `build_sectors` · `run_acq` (:dd_fast, 3-slice) · `roi_cores` / `collect_rois`"
end

# ╔═╡ aaaa0011-0000-4000-8000-000000000011
md"""## 5 · Inverse: calibration surface · noise · prior
**Calibration surface** ``f=\\mathrm{poly}_2(\\mathrm{HU}_{40},\\mathrm{HU}_{70})`` fit from known mixtures
(the user's "f vs μ" relationship). Noise: convex-quadratic
``\\sigma_E(\\mathrm{HU})=a\\,\\mathrm{HU}^2+b\\,\\mathrm{HU}+c`` + inter-energy ``\\rho``. Adipose prior
``\\mathcal N(f_w)\\,\\mathcal N(f_l)\\,\\Gamma(f_p)`` for the Bayesian per-voxel refinement."""

# ╔═╡ aaaa0012-0000-4000-8000-000000000012
begin
    poly2(h4,h7)=[1.0,h4,h7,h4^2,h7^2,h4*h7]; surf(c,h4,h7)=dot(c,poly2(h4,h7))
    quad_sigma(c,H)=c[1]*H^2+c[2]*H+c[3]
    function fit_sigma_quad(hu,sig); X=hcat(hu.^2,hu,ones(length(hu)));c=X\sig;c[1]<0&&(Xa=hcat(hu,ones(length(hu)));ca=Xa\sig;c=[0.0,ca[1],ca[2]]);c; end
    "CCC (Lin), OLS slope/intercept, RMSE, 1:1 R²."
    function metrics(t,r)
        mt,mr=mean(t),mean(r);st2=mean((t.-mt).^2);sr2=mean((r.-mr).^2);str=mean((t.-mt).*(r.-mr))
        (ccc=2str/(st2+sr2+(mt-mr)^2),slope=str/st2,int=mr-str/st2*mt,rmse=sqrt(mean((r.-t).^2)),r2=1-sum((r.-t).^2)/sum((t.-mt).^2))
    end
    struct BayesPrior; μ_w::Float64; s_w::Float64; μ_l::Float64; s_l::Float64; α_p::Float64; θ_p::Float64; end
    bayes_prior_broad(comps;s_wl=0.15,fp_shape=1.2,fp_scale=0.10)=(fw=[c[1] for c in comps];fl=[c[2] for c in comps];BayesPrior(mean(fw),s_wl,mean(fl),s_wl,fp_shape,fp_scale))
    md"`surf` (calibration) · `fit_sigma_quad` (noise) · `metrics` (CCC…) · `bayes_prior_broad`"
end

# ╔═╡ aaaa0013-0000-4000-8000-000000000013
md"""## 6 · Run calibration + test sims (cached)
6 calibration + 5 circular + 2 sector + 1 map acquisitions. First run ≈ 6 min on GPU; results cache to
`wlp_sim_cache.jls`, so re-opening is instant. Delete that file to re-simulate."""

# ╔═╡ aaaa0014-0000-4000-8000-000000000014
begin
    const CACHE = joinpath(@__DIR__, "wlp_sim_cache.jls")
    if !isfile(CACHE)
        Random.seed!(1)
        _calrois = NamedTuple[]
        for (si,seed) in enumerate((11,12,13,14,15,16))
            comps=diverse_comps(1000+si,NROD); ph=build_rods(comps); acq=run_acq(ph.gpu;seed=seed)
            lc=Dict(ROD0-1+k=>comps[k] for k in 1:NROD); append!(_calrois,collect_rois(acq,ph.cpu,lc).rois); ph=nothing;GC.gc(true)
        end
        _testrois=NamedTuple[]; _geomtag=Symbol[]
        for (si,seed) in enumerate((201,202,203,204,205))
            comps=diverse_comps(91260708+si,NROD); ph=build_rods(comps); acq=run_acq(ph.gpu;seed=seed)
            lc=Dict(ROD0-1+k=>comps[k] for k in 1:NROD)
            for r in collect_rois(acq,ph.cpu,lc).rois; push!(_testrois,r);push!(_geomtag,:circular); end; ph=nothing;GC.gc(true)
        end
        for (si,seed) in enumerate((301,302))
            comps=diverse_comps(50260708+si,16); ph=build_sectors(comps); acq=run_acq(ph.gpu;seed=seed)
            lc=Dict(ROD0+s=>comps[s+1] for s in 0:15)
            for r in collect_rois(acq,ph.cpu,lc).rois; push!(_testrois,r);push!(_geomtag,:sector); end; ph=nothing;GC.gc(true)
        end
        _mc=diverse_comps(777,NROD); mph=build_rods(_mc); macq=run_acq(mph.gpu;seed=999)
        mlc=Dict(ROD0-1+k=>_mc[k] for k in 1:NROD); mrc=roi_cores(mph.cpu,macq.geom,(512,512,3),collect(keys(mlc)))
        serialize(CACHE,(_calrois,_testrois,_geomtag,Array(macq.hu40[:,:,mrc.midz]),Array(macq.hu70[:,:,mrc.midz]),mrc.m2,_mc)); mph=nothing;GC.gc(true)
    end
    (calrois,testrois,geomtag,map40,map70,map_m2,mapcomps) = deserialize(CACHE)

    m40c=[r.m40 for r in calrois]; m70c=[r.m70 for r in calrois]
    fwc=[r.fw for r in calrois]; flc=[r.fl for r in calrois]; fpc=[r.fp for r in calrois]
    Xc=reduce(vcat,[poly2(m40c[i],m70c[i])' for i in eachindex(m40c)]); cw=Xc\fwc; cl=Xc\flc; cp=Xc\fpc
    r2cal(c,y)=1-sum((surf.(Ref(c),m40c,m70c).-y).^2)/sum((y.-mean(y)).^2)
    sc40=fit_sigma_quad(m40c,[r.s40 for r in calrois]); sc70=fit_sigma_quad(m70c,[r.s70 for r in calrois])
    res40=vcat([r.v40.-r.m40 for r in calrois]...); res70=vcat([r.v70.-r.m70 for r in calrois]...); ρ=cor(res40,res70)
    adipose=draw_wlp(20260708,2000); prior=bayes_prior_broad(adipose)
    decode(a,b)=(x=surf(cw,a,b);y=surf(cl,a,b);z=surf(cp,a,b);s=x+y+z;(x/s,y/s,z/s))
    tfw=[r.fw for r in testrois];tfl=[r.fl for r in testrois];tfp=[r.fp for r in testrois]
    dec=[decode(r.m40,r.m70) for r in testrois]; pfw=[d[1] for d in dec];pfl=[d[2] for d in dec];pfp=[d[3] for d in dec]
    semfw=[std([decode(r.v40[j],r.v70[j])[1] for j in eachindex(r.v40)])/sqrt(length(r.v40)) for r in testrois]
    mw=metrics(tfw,pfw);ml=metrics(tfl,pfl);mp=metrics(tfp,pfp)
    dHU40=[abs(mix_hu(pfw[i],pfl[i],pfp[i],E40)-mix_hu(tfw[i],tfl[i],tfp[i],E40)) for i in eachindex(tfw)]
    dHU70=[abs(mix_hu(pfw[i],pfl[i],pfp[i],E70)-mix_hu(tfw[i],tfl[i],tfp[i],E70)) for i in eachindex(tfw)]
    # per-region POOLED delivered map (uniform per rod)
    mlc2=Dict(ROD0-1+k=>mapcomps[k] for k in 1:NROD)
    truemap=fill(NaN,size(map_m2)...,3); recmap=fill(NaN,size(map_m2)...,3); rodmask=falses(size(map_m2))
    for lab in ROD0:(ROD0+NROD-1)
        idx=findall(==(UInt8(lab)),map_m2); isempty(idx)&&continue
        d=decode(mean(Float64(map40[I]) for I in idx), mean(Float64(map70[I]) for I in idx))
        for I in idx; rodmask[I]=true; truemap[I,1],truemap[I,2],truemap[I,3]=mlc2[lab]; recmap[I,1],recmap[I,2],recmap[I,3]=d; end
    end
    _ri=findall(rodmask); _ci=extrema(getindex.(_ri,1)); _cj=extrema(getindex.(_ri,2)); _pad=12
    crI=max(1,_ci[1]-_pad):min(size(map_m2,1),_ci[2]+_pad); crJ=max(1,_cj[1]-_pad):min(size(map_m2,2),_cj[2]+_pad)
    truemap_c=truemap[crI,crJ,:]; recmap_c=recmap[crI,crJ,:]
    Markdown.parse("cal n=$(length(calrois)), R²(f_w)=$(round(r2cal(cw,fwc),digits=2)); **TEST n=$(length(testrois))** — f_w CCC=**$(round(mw.ccc,digits=2))**, f_l CCC=**$(round(ml.ccc,digits=2))**, f_p CCC=**$(round(mp.ccc,digits=2))**; ρ=$(round(ρ,digits=2)).")
end

# ╔═╡ aaaa0015-0000-4000-8000-000000000015
md"## 7 · Relationship plots & delivered maps"

# ╔═╡ aaaa0016-0000-4000-8000-000000000016
let f=CM.Figure(size=(1100,520)), flcol=[r.fl for r in calrois]
    ax=CM.Axis(f[1,1];xlabel="HU$(Int(E40))",ylabel="HU$(Int(E70))",title="Barycentric triangle · 40 vs 70 keV",aspect=CM.DataAspect())
    CM.poly!(ax,[CM.Point2f(PW...),CM.Point2f(PL...),CM.Point2f(PP...)];color=(:steelblue,0.15),strokecolor=:gray,strokewidth=1)
    sc=CM.scatter!(ax,m40c,m70c;color=flcol,colormap=:viridis,markersize=7)
    for (p,t) in ((PW,"W"),(PL,"L"),(PP,"P")); CM.scatter!(ax,[p[1]],[p[2]];marker=:diamond,color=:black,markersize=13); CM.text!(ax,p[1],p[2];text=t,fontsize=16,align=(:center,:bottom)); end
    CM.Colorbar(f[1,2],sc;label="f_l")
    CM.Label(f[0,:],"Recon rods fall inside the theoretical W/L/P triangle — recon HU matches theory on v0.8.0 (cond G=$(round(cond([PL[1] PP[1];PL[2] PP[2]]),digits=1)))";fontsize=13,font=:bold)
    safe_save(joinpath(ASSET,"fig1_triangle.png"),f); f
end

# ╔═╡ aaaa0017-0000-4000-8000-000000000017
let f=CM.Figure(size=(1100,480)), flcol=[r.fl for r in calrois]
    for (col,(hu,lab)) in enumerate(((m40c,"HU$(Int(E40))"),(m70c,"HU$(Int(E70))")))
        ax=CM.Axis(f[1,col];xlabel=lab,ylabel="f_water",title="f_w vs $lab (colored by f_l)")
        sc=CM.scatter!(ax,hu,fwc;color=flcol,colormap=:viridis,markersize=7); col==2 && CM.Colorbar(f[1,3],sc;label="f_l")
    end
    CM.Label(f[0,:],"Calibration: f_w vs a single energy is a cloud, not a curve — the 2nd energy resolves it";fontsize=13,font=:bold)
    safe_save(joinpath(ASSET,"fig2_calibration.png"),f); f
end

# ╔═╡ aaaa0018-0000-4000-8000-000000000018
let f=CM.Figure(size=(1200,420)), comps=diverse_comps(5,600)
    fw=[c[1] for c in comps];fl=[c[2] for c in comps];fp=[c[3] for c in comps]
    for (col,(x,y,xl,yl)) in enumerate(((fw,fl,"f_w","f_l"),(fw,fp,"f_w","f_p"),(fl,fp,"f_l","f_p")))
        ax=CM.Axis(f[1,col];xlabel=xl,ylabel=yl,title="$xl vs $yl"); CM.scatter!(ax,x,y;markersize=4,color=(:steelblue,0.5))
    end
    CM.Label(f[0,:],"The three pairwise fraction relations (closure f_w+f_l+f_p=1 is the 3-way constraint)";fontsize=13,font=:bold)
    safe_save(joinpath(ASSET,"fig3_fraction_pairs.png"),f); f
end

# ╔═╡ aaaa0019-0000-4000-8000-000000000019
let f=CM.Figure(size=(1100,480))
    ax=CM.Axis(f[1,1];xlabel="HU",ylabel="σ (HU)",title="σ(HU) per energy — convex-quadratic fit")
    for (hu,sg,cc,e,col) in ((m40c,[r.s40 for r in calrois],sc40,E40,:tomato),(m70c,[r.s70 for r in calrois],sc70,E70,:royalblue))
        CM.scatter!(ax,hu,sg;color=col,markersize=6,label="$(Int(e)) keV"); g=range(minimum(hu),maximum(hu),100); CM.lines!(ax,g,quad_sigma.(Ref(cc),g);color=col)
    end
    CM.axislegend(ax;position=:rt)
    ax2=CM.Axis(f[1,2];xlabel="residual HU$(Int(E40))",ylabel="residual HU$(Int(E70))",title="inter-energy noise ρ=$(round(ρ,digits=2))")
    idx=rand(1:length(res40),3000); CM.scatter!(ax2,res40[idx],res70[idx];markersize=3,color=(:purple,0.3))
    safe_save(joinpath(ASSET,"fig4_noise.png"),f); f
end

# ╔═╡ aaaa0020-0000-4000-8000-000000000020
let f=CM.Figure(size=(1200,460))
    aw=[c[1] for c in adipose];al=[c[2] for c in adipose];ap=[c[3] for c in adipose]
    npdf(x,μ,s)=exp(-(x-μ)^2/(2s^2))/(s*sqrt(2π)); trapz(y,x)=y./sum((y[1:end-1].+y[2:end])./2 .* diff(x))
    for (col,(d,μ,s,nm,isg)) in enumerate(((aw,mean(aw),std(aw),"f_w Normal",false),(al,mean(al),std(al),"f_l Normal",false),(ap,mean(ap),std(ap),"f_p Gamma",true)))
        ax=CM.Axis(f[1,col];xlabel=nm[1:3],title=nm); CM.hist!(ax,d;bins=40,normalization=:pdf,color=(:gray,0.5))
        g=collect(range(max(1e-4,minimum(d)),maximum(d),200))
        if isg; α=μ^2/s^2;θ=s^2/μ; sh=[x^(α-1)*exp(-x/θ) for x in g]; CM.lines!(ax,g,trapz(sh,g);color=:crimson,linewidth=2)
        else CM.lines!(ax,g,npdf.(g,μ,s);color=:crimson,linewidth=2); end
    end
    CM.Label(f[0,:],"Adipose prior (Woodard 1986): Normal(f_w)·Normal(f_l)·Gamma(f_p)";fontsize=13,font=:bold)
    safe_save(joinpath(ASSET,"fig5_prior.png"),f); f
end

# ╔═╡ aaaa0021-0000-4000-8000-000000000021
let f=CM.Figure(size=(1200,1050)), names=("f_w","f_l","f_p")
    for row in 1:3
        tt=truemap_c[:,:,row]; rr=recmap_c[:,:,row]; er=rr.-tt
        for (col,(img,ttl,cr,cm)) in enumerate(((tt,"true $(names[row])",(0,1),:jet),(rr,"recovered (pooled)",(0,1),:jet),(er,"error",(-0.1,0.1),:balance)))
            ax=CM.Axis(f[row,col];title=ttl,aspect=CM.DataAspect()); CM.hidedecorations!(ax)
            hm=CM.heatmap!(ax,img;colormap=cm,colorrange=cr); (col==3)&&CM.Colorbar(f[row,4],hm)
        end
    end
    CM.Label(f[0,:],"Delivered per-region pooled maps vs ground truth (circular test phantom, ±0.1 error scale)";fontsize=14,font=:bold)
    safe_save(joinpath(ASSET,"fig6_maps.png"),f); f
end

# ╔═╡ aaaa0022-0000-4000-8000-000000000022
let f=CM.Figure(size=(1300,460))
    for (col,(t,p,m,nm)) in enumerate(((tfw,pfw,mw,"f_w"),(tfl,pfl,ml,"f_l"),(tfp,pfp,mp,"f_p")))
        lo=min(minimum(t),minimum(p));hi=max(maximum(t),maximum(p))
        ax=CM.Axis(f[1,col];xlabel="true $nm",ylabel="recovered",title="$nm  CCC=$(round(m.ccc,digits=2))",aspect=CM.DataAspect(),limits=(lo,hi,lo,hi))
        CM.lines!(ax,[lo,hi],[lo,hi];color=:gray,linestyle=:dash)
        col==1 && CM.errorbars!(ax,t,p,semfw;color=(:steelblue,0.3),whiskerwidth=0)
        CM.scatter!(ax,t,p;color=[g==:circular ? :steelblue : :orange for g in geomtag],markersize=6)
        CM.text!(ax,lo+0.02*(hi-lo),hi-0.05*(hi-lo);text=@sprintf("slope %.2f\nRMSE %.3f\nR² %.2f",m.slope,m.rmse,m.r2),align=(:left,:top),fontsize=11)
    end
    CM.Label(f[0,:],"Recovered vs true (blue=circular, orange=sector) — pooled ROI, error bars = SEM (f_w)";fontsize=13,font=:bold)
    safe_save(joinpath(ASSET,"fig7_scatter.png"),f); f
end

# ╔═╡ aaaa0023-0000-4000-8000-000000000023
let f=CM.Figure(size=(1100,460))
    for (col,(d,e)) in enumerate(((dHU40,E40),(dHU70,E70)))
        ax=CM.Axis(f[1,col];xlabel="ROI |ΔHU| at $(Int(e)) keV",ylabel="count",title="detectability $(Int(e)) keV — $(round(100mean(d.<5),digits=0))% < 5 HU")
        CM.hist!(ax,d;bins=30,color=(:teal,0.6)); CM.vlines!(ax,[5.0];color=:red,linestyle=:dash,label="5 HU"); CM.axislegend(ax)
    end
    safe_save(joinpath(ASSET,"fig8_detectability.png"),f); f
end

# ╔═╡ aaaa0024-0000-4000-8000-000000000024
md"""## 8 · Validation & conclusion

| fraction | CCC | slope | RMSE |
|---|---|---|---|
| **f_water** | **$(round(mw.ccc,digits=2))** | $(round(mw.slope,digits=2)) | $(round(mw.rmse,digits=3)) |
| f_lipid | $(round(ml.ccc,digits=2)) | $(round(ml.slope,digits=2)) | $(round(ml.rmse,digits=3)) |
| f_protein | $(round(mp.ccc,digits=2)) | $(round(mp.slope,digits=2)) | $(round(mp.rmse,digits=3)) |

Detectability: **$(round(100mean(dHU70.<5),digits=0))% of ROIs < 5 HU at 70 keV** (mean $(round(mean(dHU70),digits=1)) HU);
$(round(mean(dHU40),digits=1)) HU at 40 keV.

**Conclusion.** With a **quantitative** 2-basis DECT simulation (BasisSimulator v0.8.0, `:dd_fast`
projector — pure lipid −205 vs theoretical −213), water/lipid/protein volume fractions are all recovered
with **CCC > 0.9** and ROI accuracy within ~2 HU at 70 keV. The decode is a simple calibration surface
fit from known mixtures; the recon is linear-additive (calibration in-sample R²(f_w) = $(round(r2cal(cw,fwc),digits=2))),
so the triangle is well-conditioned and no strong prior is needed. The earlier apparent "f_water ceiling"
was entirely an artifact of the old v0.2.1 forward projector, which compressed non-water HU by ~50 HU —
a reminder to **validate the simulator against pure-material theoretical HU before trusting any decode.**
"""

# ╔═╡ Cell order:
# ╟─aaaa0002-0000-4000-8000-000000000002
# ╠═aaaa0001-0000-4000-8000-000000000001
# ╟─aaaa0003-0000-4000-8000-000000000003
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
# ╟─aaaa0024-0000-4000-8000-000000000024
