# ============================================================
# Methods of Kinetic Theory
# Double dispatches for 1> Galerkin and 2> collocation methods
# Multiple dispatches for different particle velocity settings
# ============================================================


export uq_moments_conserve,
    uq_maxwellian,
    uq_prim_conserve,
    uq_conserve_prim,
    uq_conserve_prim!,
    uq_sound_speed,
    uq_vhs_collision_time,
    uq_aap_hs_collision_time,
    uq_aap_hs_prim


"""
Calculate conservative moments from distribution function

"""

# ------------------------------------------------------------
# Single-component gas
# ------------------------------------------------------------
function uq_moments_conserve(
    f::AbstractArray{<:AbstractFloat,2},
    u::AbstractArray{<:AbstractFloat,1},
    ω::AbstractArray{<:AbstractFloat,1},
)

    w = zeros(eltype(f), 3, axes(f, 2))

    for j in axes(w, 2)
        w[:, j] .= Kinetic.moments_conserve(f[:, j], u, ω)
    end

    return w

end

function uq_moments_conserve(
    h::AbstractArray{<:AbstractFloat,2},
    b::AbstractArray{<:AbstractFloat,2},
    u::AbstractArray{<:AbstractFloat,1},
    ω::AbstractArray{<:AbstractFloat,1},
)

    w = zeros(eltype(f), 3, axes(h, 2))

    for j in axes(w, 2)
        w[:, j] .= Kinetic.moments_conserve(h[:, j], b[:, j], u, ω)
    end

    return w

end

function uq_moments_conserve(
    f::AbstractArray{<:AbstractFloat,3},
    u::AbstractArray{<:AbstractFloat,2},
    v::AbstractArray{<:AbstractFloat,2},
    ω::AbstractArray{<:AbstractFloat,2},
)

    w = zeros(eltype(f), 4, axes(h, 3))

    for j in axes(w, 2)
        w[:, j] .= Kinetic.moments_conserve(f[:, :, j], u, v, ω)
    end

    return w

end

function uq_moments_conserve(
    h::AbstractArray{<:AbstractFloat,3},
    b::AbstractArray{<:AbstractFloat,3},
    u::AbstractArray{<:AbstractFloat,2},
    v::AbstractArray{<:AbstractFloat,2},
    ω::AbstractArray{<:AbstractFloat,2},
)

    w = zeros(eltype(f), 4, axes(h, 3))

    for j in axes(w, 2)
        w[:, j] .= Kinetic.moments_conserve(h[:, :, j], b[:, :, j], u, v, ω)
    end

    return w

end


# ------------------------------------------------------------
# Multi-component gas
# ------------------------------------------------------------
function uq_moments_conserve(
    h0::AbstractArray{<:AbstractFloat,3},
    h1::AbstractArray{<:AbstractFloat,3},
    h2::AbstractArray{<:AbstractFloat,3},
    h3::AbstractArray{<:AbstractFloat,3},
    u::AbstractArray{<:AbstractFloat,2},
    ω::AbstractArray{<:AbstractFloat,2},
)

    w = zeros(5, size(h0, 2), size(h0, 3))

    for k in axes(w, 3)
        for j in axes(w, 2)
            w[1, j, k] = Kinetic.moments_conserve(
                h0[:, j, k],
                h1[:, j, k],
                h2[:, j, k],
                h3[:, j, k],
                u[:, k],
                ω[:, k],
            )
        end
    end

    return w

end


"""
Calculate equilibrium distribution

"""

# ------------------------------------------------------------
# Single-component gas
# ------------------------------------------------------------
function uq_maxwellian(
    uspace::AbstractArray{<:AbstractFloat,1},
    prim::Array{<:AbstractFloat,2},
    uq::AbstractUQ,
)

    if size(prim, 2) == uq.nr + 1

        primRan = chaos_ran(prim, 2, uq)

        MRan = zeros(axes(uspace, 1), axes(primRan, 2))
        for j in axes(MRan, 2)
            MRan[:, j] .= Kinetic.maxwellian(uspace, primRan[:, j])
        end

        M = ran_chaos(MRan, 2, uq)

        return M

    elseif size(prim, 2) == uq.op.quad.Nquad

        M = zeros(axes(uspace, 1), axes(prim, 2))
        for j in axes(M, 2)
            M[:, j] .= Kinetic.maxwellian(uspace, prim[:, j])
        end

        return M

    else

        throw(DimensionMismatch("inconsistent random domain size in settings and solutions"))

    end

end

function uq_maxwellian(
    u::AbstractArray{<:AbstractFloat,2},
    v::AbstractArray{<:AbstractFloat,2},
    prim::Array{<:AbstractFloat,2},
    uq::AbstractUQ,
)

    if size(prim, 2) == uq.nr + 1

        primRan = chaos_ran(prim, 2, uq)

        MRan = zeros((axes(u)..., axes(primRan, 2)))
        for k in axes(MRan, 3)
            MRan[:, :, k] .= Kinetic.maxwellian(u, v, primRan[:, k])
        end

        M = ran_chaos(MRan, 3, uq)

        return M

    elseif size(prim, 2) == uq.op.quad.Nquad

        M = zeros((axes(u)..., axes(prim, 2)))
        for k in axes(M, 3)
            M[:, :, k] .= Kinetic.maxwellian(u, v, prim[:, k])
        end

        return M

    else

        throw(DimensionMismatch("inconsistent random domain size in settings and solutions"))

    end

end

function uq_maxwellian(
    u::AbstractArray{<:AbstractFloat,2},
    v::AbstractArray{<:AbstractFloat,2},
    prim::Array{<:AbstractFloat,2},
    uq::AbstractUQ,
    inK::Real,
)

    if size(prim, 2) == uq.nr + 1

        primRan = chaos_ran(prim, 2, uq)

        HRan = zeros((axes(u)..., axes(primRan, 2)))
        BRan = similar(HRan)
        for k in axes(HRan, 3)
            HRan[:, :, k] .= Kinetic.maxwellian(u, v, primRan[:, k])
            BRan[:, :, k] .= HRan[:, :, k] .* inK ./ (2.0 * primRan[end, k])
        end

        H = ran_chaos(HRan, 3, uq)
        B = ran_chaos(BRan, 3, uq)

        return H, B

    elseif size(prim, 2) == uq.op.quad.Nquad

        H = zeros((axes(u)..., axes(prim, 2)))
        B = similar(H)
        for k in axes(H, 3)
            H[:, :, k] .= Kinetic.maxwellian(u, v, prim[:, k])
            B[:, :, k] .= H[:, :, k] .* inK ./ (2.0 * prim[end, k])
        end

        return H, B

    else

        throw(DimensionMismatch("inconsistent random domain size in settings and solutions"))

    end

end


# ------------------------------------------------------------
# Multi-component plasma
# ------------------------------------------------------------
function uq_maxwellian(
    uspace::AbstractArray{<:AbstractFloat,2},
    prim::Array{<:AbstractFloat,3},
    uq::AbstractUQ,
)

    if size(prim, 2) == uq.nr + 1

        primRan = chaos_ran(prim, 2, uq)

        Mv = zeros(uq.op.quad.Nquad, 2)
        Mw = similar(Mv)
        for k in axes(Mv, 2)
            for j in axes(Mv, 1)
                Mv[j, k] = primRan[3, j, k]^2 + 0.5 / primRan[end, j, k]
                Mw[j, k] = primRan[4, j, k]^2 + 0.5 / primRan[end, j, k]
            end
        end

        H0Ran =
            zeros(axes(uspace, 1), 1:length(uq.op.quad.nodes), axes(prim, 3))
        H1Ran = similar(H0Ran)
        H2Ran = similar(H0Ran)
        H3Ran = similar(H0Ran)
        for k in axes(H0Ran, 3)
            for j in axes(H0Ran, 2)
                H0Ran[:, j, k] .= Kinetic.maxwellian(
                    uspace[:, k],
                    primRan[1, j, k],
                    primRan[2, j, k],
                    primRan[5, j, k],
                )
                H1Ran[:, j, k] .= primRan[3, j, k] .* H0Ran[:, j, k]
                H2Ran[:, j, k] .= primRan[4, j, k] .* H0Ran[:, j, k]
                H3Ran[:, j, k] .= (Mv[j, k] + Mw[j, k]) .* H0Ran[:, j, k]
            end
        end

        H0 = ran_chaos(H0Ran, 2, uq)
        H1 = ran_chaos(H1Ran, 2, uq)
        H2 = ran_chaos(H2Ran, 2, uq)
        H3 = ran_chaos(H3Ran, 2, uq)

        return H0, H1, H2, H3

    elseif size(prim, 2) == uq.op.quad.Nquad

        Mv = zeros(axes(prim, 2), 2)
        Mw = similar(Mv)
        for k in axes(Mv, 2)
            for j in axes(Mv, 1)
                Mv[j, k] = prim[3, j, k]^2 + 0.5 / prim[end, j, k]
                Mw[j, k] = prim[4, j, k]^2 + 0.5 / prim[end, j, k]
            end
        end

        H0 = zeros(axes(uspace, 1), axes(prim, 2), axes(prim, 3))
        H1 = similar(H0)
        H2 = similar(H0)
        H3 = similar(H0)
        for k in axes(H0, 3)
            for j in axes(H0, 2)
                H0[:, j, k] .= Kinetic.maxwellian(
                    uspace[:, k],
                    prim[1, j, k],
                    prim[2, j, k],
                    prim[5, j, k],
                )
                H1[:, j, k] .= prim[3, j, k] .* H0[:, j, k]
                H2[:, j, k] .= prim[4, j, k] .* H0[:, j, k]
                H3[:, j, k] .= (Mv[j, k] + Mw[j, k]) .* H0[:, j, k]
            end
        end

        return H0, H1, H2, H3

    else

        throw(DimensionMismatch("inconsistent random domain size in settings and solutions"))

    end

end


"""
Calculate conservative/primitive variables

"""

function uq_prim_conserve(
    prim::Array{<:AbstractFloat,2},
    gamma::Real,
    uq::AbstractUQ,
)

    if size(prim, 2) == uq.nr + 1

        primRan = chaos_ran(prim, 2, uq)

        wRan = similar(primRan)
        for j in axes(wRan, 2)
            wRan[:, j] .= Kinetic.prim_conserve(primRan[:, j], gamma)
        end

        wChaos = zeros(axes(prim))
        for i in axes(wChaos, 1)
            wChaos[i, :] .= ran_chaos(wRan[i, :], uq)
        end

        return wChaos

    elseif size(prim, 2) == uq.op.quad.Nquad

        wRan = similar(prim)
        for j in axes(wRan, 2)
            wRan[:, j] .= Kinetic.prim_conserve(prim[:, j], gamma)
        end

        return wRan

    else

        throw(DimensionMismatch("inconsistent random domain size in settings and solutions"))

    end

end

function uq_prim_conserve(
    prim::Array{<:AbstractFloat,3},
    gamma::Real,
    uq::AbstractUQ,
)

    if size(prim, 2) == uq.nr + 1

        primRan = chaos_ran(prim, 2, uq)

        wRan = similar(primRan)
        for k in axes(wRan, 3)
            for j in axes(wRan, 2)
                wRan[:, j, k] .= Kinetic.prim_conserve(primRan[:, j, k], gamma)
            end
        end

        wChaos = zeros(axes(prim))
        for k in axes(wChaos, 3)
            for i in axes(wChaos, 1)
                wChaos[i, :, k] .= ran_chaos(wRan[i, :, k], uq)
            end
        end

        return wChaos

    elseif size(prim, 2) == uq.op.quad.Nquad

        wRan = similar(prim)
        for k in axes(wRan, 3)
            for j in axes(wRan, 2)
                wRan[:, j, k] .= Kinetic.prim_conserve(prim[:, j, k], gamma)
            end
        end

        return wRan

    else

        throw(DimensionMismatch("inconsistent random domain size in settings and solutions"))

    end

end


function uq_conserve_prim(
    w::Array{<:AbstractFloat,2},
    gamma::Real,
    uq::AbstractUQ,
)

    if size(w, 2) == uq.nr + 1

        wRan = chaos_ran(w, 2, uq)

        primRan = similar(wRan)
        for j in axes(primRan, 2)
            primRan[:, j] .= Kinetic.conserve_prim(wRan[:, j], gamma)
        end

        primChaos = zeros(axes(w))
        for i in axes(primChaos, 1)
            primChaos[i, :] .= ran_chaos(primRan[i, :], uq)
        end

        return primChaos

    elseif size(w, 2) == uq.op.quad.Nquad

        primRan = similar(w)
        for j in axes(primRan, 2)
            primRan[:, j] .= Kinetic.conserve_prim(w[:, j], gamma)
        end

        return primRan

    else

        throw(DimensionMismatch("inconsistent random domain size in settings and solutions"))

    end

end

function uq_conserve_prim(
    w::Array{<:AbstractFloat,3},
    gamma::Real,
    uq::AbstractUQ,
)

    if size(w, 2) == uq.nr + 1

        wRan = chaos_ran(w, 2, uq)

        primRan = similar(wRan)
        for k in axes(primRan, 3)
            for j in axes(primRan, 2)
                primRan[:, j, k] .= Kinetic.conserve_prim(wRan[:, j, k], gamma)
            end
        end

        primChaos = zeros(axes(w))
        for k in axes(primChaos, 3)
            for i in axes(primChaos, 1)
                primChaos[i, :, k] .= ran_chaos(primRan[i, :, k], uq)
            end
        end

        return primChaos

    elseif size(w, 2) == uq.op.quad.Nquad

        primRan = similar(w)
        for k in axes(primRan, 3)
            for j in axes(primRan, 2)
                primRan[:, j, k] .= Kinetic.conserve_prim(w[:, j, k], gamma)
            end
        end

        return primRan

    else

        throw(DimensionMismatch("inconsistent random domain size in settings and solutions"))

    end

end

function uq_conserve_prim!(sol::Solution1D1F, γ::Real, uq::AbstractUQ)

    for i in eachindex(sol.w)
        sol.prim[i] .= uq_conserve_prim(sol.w[i], γ, uq)
    end

end


"""
Calculate speed of sound

"""

function uq_sound_speed(
    prim::Array{<:AbstractFloat,2},
    gamma::Real,
    uq::AbstractUQ,
)

    if size(prim, 2) == uq.nr + 1

        primRan = chaos_ran(prim, 2, uq)

        sosRan = zeros(uq.op.quad.Nquad)
        for j in eachindex(sosRan)
            sosRan[j] = Kinetic.sound_speed(primRan[end, j], gamma)
        end

        return ran_chaos(sosRan, uq)

    elseif size(prim, 2) == uq.op.quad.Nquad

        sosRan = zeros(axes(prim, 2))
        for j in eachindex(sosRan)
            sosRan[j] = Kinetic.sound_speed(prim[end, j], gamma)
        end

        return sosRan

    else

        throw(DimensionMismatch("inconsistent random domain size in settings and solutions"))

    end

end

function uq_sound_speed(
    prim::Array{<:AbstractFloat,3},
    gamma::Real,
    uq::AbstractUQ,
)

    if size(prim, 2) == uq.nr + 1

        primRan = chaos_ran(prim, 2, uq)

        sosRan = zeros(uq.op.quad.Nquad)
        for j in eachindex(sosRan)
            sosRan[j] = max(
                Kinetic.sound_speed(primRan[end, j, 1], gamma),
                Kinetic.sound_speed(primRan[end, j, 2], gamma),
            )
        end

        return ran_chaos(sosRan, uq)

    elseif size(prim, 2) == uq.op.quad.Nquad

        sosRan = zeros(axes(prim, 2))
        for j in eachindex(sosRan)
            sosRan[j] = max(
                Kinetic.sound_speed(prim[end, j, 1], gamma),
                Kinetic.sound_speed(prim[end, j, 2], gamma),
            )
        end

        return sosRan

    else

        throw(DimensionMismatch("inconsistent random domain size in settings and solutions"))

    end

end


"""
Calculate collision time

"""

function uq_vhs_collision_time(
    prim::Array{<:AbstractFloat,2},
    muRef::Real,
    omega::Real,
    uq::AbstractUQ,
)

    if size(prim, 2) == uq.nr + 1

        primRan = chaos_ran(prim, 2, uq)

        tauRan = zeros(uq.op.quad.Nquad)
        for i in eachindex(tauRan)
            tauRan[i] = Kinetic.vhs_collision_time(primRan[:, i], muRef, omega)
        end

        return ran_chaos(tauRan, uq)

    elseif size(prim, 2) == uq.op.quad.Nquad

        tau = zeros(uq.op.quad.Nquad)
        for i in eachindex(tau)
            tau[i] = Kinetic.vhs_collision_time(prim[:, i], muRef, omega)
        end

        return tau

    else

        throw(DimensionMismatch("inconsistent random domain size in settings and solutions"))

    end

end

function uq_vhs_collision_time(
    prim::Array{<:AbstractFloat,2},
    muRef::Array{<:AbstractFloat,1},
    omega::Real,
    uq::AbstractUQ,
)

    if size(prim, 2) == uq.nr + 1

        primRan = chaos_ran(prim, 2, uq)
        muRan = chaos_ran(muRef, uq)

        tauRan = zeros(uq.op.quad.Nquad)
        for i in eachindex(tauRan)
            tauRan[i] =
                Kinetic.vhs_collision_time(primRan[:, i], muRan[i], omega)
        end

        return ran_chaos(tauRan, uq)

    elseif size(prim, 2) == uq.op.quad.Nquad

        tau = zeros(uq.op.quad.Nquad)
        for i in eachindex(tau)
            tau[i] = Kinetic.vhs_collision_time(prim[:, i], muRef[i], omega)
        end

        return tau

    else

        throw(DimensionMismatch("inconsistent random domain size in settings and solutions"))

    end

end

function uq_vhs_collision_time(
    sol::AbstractSolution1D,
    muRef::Union{Real,Array{<:AbstractFloat,1}},
    omega::Real,
    uq::AbstractUQ,
)

    tau = [
        uq_vhs_collision_time(sol.prim[i], muRef, omega, uq)
        for i in eachindex(sol.prim)
    ]

end

function uq_vhs_collision_time(
    sol::AbstractSolution2D,
    muRef::Union{Real,Array{<:AbstractFloat,1}},
    omega::Real,
    uq::AbstractUQ,
)

    tau = [
        uq_vhs_collision_time(sol.prim[i, j], muRef, omega, uq)
        for i in axes(sol.prim, 1), j in axes(sol.prim, 2)
    ]

end


"""
Calculate mixed collision time in AAP model

"""

function uq_aap_hs_collision_time(
    P::Array{<:AbstractFloat,3},
    mi::Real,
    ni::Real,
    me::Real,
    ne::Real,
    kn::Real,
    uq::AbstractUQ,
)

    if size(P, 2) == uq.nr + 1
        prim = deepcopy(P[:, 1, :])
        τ = Kinetic.aap_hs_collision_time(prim, mi, ni, me, ne, kn)
        return τ
    elseif size(P, 2) == uq.op.quad.Nquad
        prim = deepcopy(P[:, end÷2+1, :])
        τ = Kinetic.aap_hs_collision_time(prim, mi, ni, me, ne, kn)
        return τ
    else
        throw(DimensionMismatch("inconsistent random domain size in settings and solutions"))
    end

end


"""
Calculate mixed primitive variables

"""

function uq_aap_hs_prim(
    prim::Array{<:AbstractFloat,3},
    tau::Array{<:AbstractFloat,1},
    mi::Real,
    ni::Real,
    me::Real,
    ne::Real,
    kn::Real,
    uq::AbstractUQ,
)

    if size(prim, 2) == uq.nr + 1

        primRan = chaos_ran(prim, 2, uq)

        # hard-sphere molecule
        mixPrimRan = similar(primRan)
        for j in axes(mixPrimRan, 2)
            mixPrimRan[:, j, :] .=
                Kinetic.aap_hs_prim(primRan[:, j, :], tau, mi, ni, me, ne, kn)
        end

        mixPrimChaos = similar(prim)
        for k in axes(mixPrimChaos, 3)
            for i in axes(mixPrimChaos, 1)
                mixPrimChaos[i, :, k] .= ran_chaos(mixPrimRan[i, :, k], uq)
            end
        end

        return mixPrimChaos

    elseif size(prim, 2) == uq.op.quad.Nquad

        mixPrimRan = similar(prim)
        for j in axes(mixPrimRan, 2)
            mixPrimRan[:, j, :] .=
                Kinetic.aap_hs_prim(prim[:, j, :], tau, mi, ni, me, ne, kn)
        end

        return mixPrimRan

    else

        throw(DimensionMismatch("inconsistent random domain size in settings and solutions"))

    end

end
