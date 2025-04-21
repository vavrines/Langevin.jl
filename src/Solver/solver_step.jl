"""
$(SIGNATURES)

Time stepping solver
"""
function step!(
    KS,
    uq::AbstractUQ,
    faceL,
    cell::T,
    faceR,
    p,
    coll=:bgk,
) where {T<:Union{ControlVolume1F,ControlVolume1D1F}} # 1D1F1V
    dt, dx, RES, AVG = p

    if uq.method == "galerkin"

        #--- update conservative flow variables: step 1 ---#
        # w^n
        w_old = deepcopy(cell.w)
        prim_old = deepcopy(cell.prim)

        # flux -> w^{n+1}
        @. cell.w += (faceL.fw - faceR.fw) / dx
        cell.prim .= uq_conserve_prim(cell.w, KS.gas.γ, uq)

        # locate variables on random quadrature points
        wRan = chaos_ran(cell.w, 2, uq)
        primRan = chaos_ran(cell.prim, 2, uq)

        # pure collocation settings
        #wRan =
        #    chaos_ran(cell.w, 2, uq) .+
        #    (chaos_ran(faceL.fw, 2, uq) .- chaos_ran(faceR.fw, 2, uq)) ./ dx
        #primRan = uq_conserve_prim(wRan, KS.gas.γ, uq)

        # temperature protection
        if min(minimum(primRan[1, :]), minimum(primRan[end, :])) < 0
            @warn "negative temperature update"
            #wRan = chaos_ran(w_old, 2, uq)
            #primRan = chaos_ran(prim_old, 2, uq)

            @info "filtering"
            #=
            Lnorm = ones(uq.nr+1)
            for i in eachindex(Lnorm)
                Lnorm[i] = sum(uq.op.quad.weights .* 2 .* abs.(evaluate(i-1, uq.op.quad.nodes, uq.op)))
                #Lnorm[i] = exp(sum(uq.op.quad.weights .* abs.(evaluate(i-1, uq.op.quad.nodes, uq.op))))
            end
            =#
            iter = 1
            while min(minimum(primRan[1, :]), minimum(primRan[end, :])) < 0
                filter!(cell.w, 2, iter * dt, maximum(KS.ib.wL .- KS.ib.wR), uq.op, :l2)
                #filter!(cell.w, Lnorm, 1e-3, 2, :adapt)
                wRan = chaos_ran(cell.w, 2, uq)
                primRan = chaos_ran(cell.prim, 2, uq)

                iter += 1
                iter > 4 && break
            end
        end

        cell.w .= ran_chaos(wRan, 2, uq)
        cell.prim .= ran_chaos(primRan, 2, uq)

        #--- update particle distribution function ---#
        # flux -> f^{n+1}
        #@. cell.f += (faceL.ff - faceR.ff) / dx

        fRan =
            chaos_ran(cell.f, 2, uq) .+
            (chaos_ran(faceL.ff, 2, uq) .- chaos_ran(faceR.ff, 2, uq)) ./ dx

        # source -> f^{n+1}
        tau = uq_vhs_collision_time(primRan, KS.gas.μᵣ, KS.gas.ω, uq)

        gRan = zeros(KS.vs.nu, uq.op.quad.Nquad)
        for j in axes(gRan, 2)
            gRan[:, j] .= KitBase.maxwellian(KS.vs.u, primRan[:, j])
        end

        # BGK term
        for j in axes(fRan, 2)
            @. fRan[:, j] = (fRan[:, j] + dt / tau[j] * gRan[:, j]) / (1.0 + dt / tau[j])
        end

        cell.f .= ran_chaos(fRan, 2, uq)

        #--- record residuals ---#
        @. RES += (w_old[:, 1] - cell.w[:, 1])^2
        @. AVG += abs(cell.w[:, 1])

    elseif uq.method == "collocation"

        #--- update conservative flow variables: step 1 ---#
        # w^n
        w_old = deepcopy(cell.w)
        prim_old = deepcopy(cell.prim)

        # flux -> w^{n+1}
        @. cell.w += (faceL.fw - faceR.fw) / dx
        cell.prim .= uq_conserve_prim(cell.w, KS.gas.γ, uq)

        # temperature protection
        if minimum(cell.prim[end, :]) < 0
            @warn "negative temperature update"
            cell.w .= w_old
            cell.prim .= prim_old
        end

        #--- update particle distribution function ---#
        # flux -> f^{n+1}
        @. cell.f += (faceL.ff - faceR.ff) / dx

        # source -> f^{n+1}
        tau = uq_vhs_collision_time(cell.prim, KS.gas.μᵣ, KS.gas.ω, uq)

        g = zeros(KS.vs.nu, uq.op.quad.Nquad)
        for j in axes(g, 2)
            g[:, j] .= maxwellian(KS.vs.u, cell.prim[:, j])
        end

        # BGK term
        for j in axes(cell.f, 2)
            @. cell.f[:, j] = (cell.f[:, j] + dt / tau[j] * g[:, j]) / (1.0 + dt / tau[j])
        end

        #--- record residuals ---#
        @. RES += (w_old[:, 1] - cell.w[:, 1])^2
        @. AVG += abs(cell.w[:, 1])
    end
end

"""
$(SIGNATURES)

1D2F1V
"""
function step!(
    KS,
    uq::AbstractUQ,
    faceL,
    cell::T,
    faceR,
    p,
    coll=:bgk,
) where {T<:Union{ControlVolume2F,ControlVolume1D2F}}
    dt, dx, RES, AVG = p

    w_old = deepcopy(cell.w)

    @. cell.w += (faceL.fw - faceR.fw) / dx
    cell.prim .= uq_conserve_prim(cell.w, KS.gas.γ, uq)
    primRan = chaos_ran(cell.prim, 2, uq)

    # realizability protection
    if uq.method == "galerkin"
        if min(minimum(primRan[1, :]), minimum(primRan[end, :])) < 0
            @warn "negative temperature update"
            @info "filter processing"

            iter = 1
            while min(minimum(primRan[1, :]), minimum(primRan[end, :])) < 0
                δ = maximum(abs.(KS.ib.fw(KS.ps.x0, KS.ib.p) .-
                             KS.ib.fw(KS.ps.x1, KS.ib.p)),)
                λ = adapt_filter_strength(cell.prim[1, :], iter * dt, δ, uq)
                for i in axes(cell.w, 1)
                    _u = @view cell.prim[i, :]
                    modal_filter!(_u, λ; filter=:l2)
                end
                for i in axes(cell.h, 1)
                    _h = @view cell.h[i, :]
                    _b = @view cell.b[i, :]

                    modal_filter!(_h, λ; filter=:l2)
                    modal_filter!(_b, λ; filter=:l2)
                end

                primRan = chaos_ran(cell.prim, 2, uq)

                iter += 1
                iter > 4 && break
            end
        end

        cell.prim .= ran_chaos(primRan, 2, uq)
        cell.w .= uq_prim_conserve(cell.prim, KS.gas.γ, uq)
    end

    @. cell.h += (faceL.fh - faceR.fh) / dx
    @. cell.b += (faceL.fb - faceR.fb) / dx

    hRan, bRan = begin
        if uq.method == "galerkin"
            chaos_ran(cell.h, 2, uq), chaos_ran(cell.b, 2, uq)
        else
            cell.h, cell.b
        end
    end

    # source -> f^{n+1}
    tau = uq_vhs_collision_time(primRan, KS.gas.μᵣ, KS.gas.ω, uq)

    HRan = uq_maxwellian(KS.vs.u, primRan, uq)
    BRan = uq_energy_distribution(HRan, primRan, KS.gas.K, uq)

    # BGK term
    for j in axes(hRan, 2)
        @. hRan[:, j] = (hRan[:, j] + dt / tau[j] * HRan[:, j]) / (1.0 + dt / tau[j])
        @. bRan[:, j] = (bRan[:, j] + dt / tau[j] * BRan[:, j]) / (1.0 + dt / tau[j])
    end

    if uq.method == "galerkin"
        cell.h .= ran_chaos(hRan, 2, uq)
        cell.b .= ran_chaos(bRan, 2, uq)
    end

    #--- record residuals ---#
    @. RES += (w_old[:, 1] - cell.w[:, 1])^2
    @. AVG += abs(cell.w[:, 1])
end

"""
$(SIGNATURES)

1D4F1V mixture
"""
function step!(
    KS,
    uq::AbstractUQ,
    faceL,
    cell::ControlVolume1D4F,
    faceR,
    p,
    coll=:bgk,
    isMHD=true,
)
    dt, dx, RES, AVG = p

    if uq.method == "galerkin"

        #--- update conservative flow variables: step 1 ---#
        # w^n
        w_old = deepcopy(cell.w)
        prim_old = deepcopy(cell.prim)

        # flux -> w^{n+1}
        #@. cell.w += (faceL.fw - faceR.fw) / dx
        #cell.prim .= get_primitive(cell.w, KS.gas.γ, uq)

        # locate variables on random quadrature points
        #wRan = get_ran_array(cell.w, 2, uq)
        #primRan = get_ran_array(cell.prim, 2, uq)

        # pure collocation settings
        wRan =
            chaos_ran(cell.w, 2, uq) .+
            (chaos_ran(faceL.fw, 2, uq) .- chaos_ran(faceR.fw, 2, uq)) ./ dx
        primRan = uq_conserve_prim(wRan, KS.gas.γ, uq)

        # temperature protection
        if min(minimum(primRan[5, :, 1]), minimum(primRan[5, :, 2])) < 0
            println("warning: temperature update is negative")
            wRan = chaos_ran(w_old, 2, uq)
            primRan = chaos_ran(prim_old, 2, uq)
        end

        #=
        # source -> w^{n+1}
        # DifferentialEquations.jl
        tau = get_tau(cell.prim, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1], uq)
        for j in axes(wRan, 2)
        prob = ODEProblem( mixture_source,
                    vcat(wRan[1:5,j,1], wRan[1:5,j,2]),
                    dt,
                    (tau[1], tau[2], KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1], KS.gas.γ) )
        sol = solve(prob, Rosenbrock23())

        wRan[1:5,j,1] .= sol[end][1:5]
        wRan[1:5,j,2] .= sol[end][6:10]
        for k=1:2
        primRan[:,j,k] .= KitBase.conserve_prim(wRan[:,j,k], KS.gas.γ)
        end
        end

        # explicit
        tau = get_tau(cell.prim, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1], uq)
        mprim = get_mixprim(cell.prim, tau, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1], uq)
        mw = get_conserved(mprim, KS.gas.γ, uq)
        for k=1:2
        cell.w[:,:,k] .+= (mw[:,:,k] .- w_old[:,:,k]) .* dt ./ tau[k]
        end
        cell.prim .= get_primitive(cell.w, KS.gas.γ, uq)

        wRan .= get_ran_array(cell.w, 2, uq);
        primRan .= get_ran_array(cell.prim, 2, uq);
        =#

        #--- update electromagnetic variables ---#
        # flux -> E^{n+1} & B^{n+1}
        #@. cell.E[1,:] -= dt * (faceL.femR[1,:] + faceR.femL[1,:]) / dx
        #@. cell.E[2,:] -= dt * (faceL.femR[2,:] + faceR.femL[2,:]) / dx
        #@. cell.E[3,:] -= dt * (faceL.femR[3,:] + faceR.femL[3,:]) / dx
        #@. cell.B[1,:] -= dt * (faceL.femR[4,:] + faceR.femL[4,:]) / dx
        #@. cell.B[2,:] -= dt * (faceL.femR[5,:] + faceR.femL[5,:]) / dx
        #@. cell.B[3,:] -= dt * (faceL.femR[6,:] + faceR.femL[6,:]) / dx
        @. cell.ϕ -= dt * (faceL.femR[7, :] + faceR.femL[7, :]) / dx
        @. cell.ψ -= dt * (faceL.femR[8, :] + faceR.femL[8, :]) / dx

        ERan = chaos_ran(cell.E, 2, uq)
        ERan[1, :] .-=
            dt .* (evaluatePCE(faceL.femR[1, :], uq.op.quad.nodes, uq.op) .+
             evaluatePCE(faceR.femL[1, :], uq.op.quad.nodes, uq.op)) ./ dx
        ERan[2, :] .-=
            dt .* (evaluatePCE(faceL.femR[2, :], uq.op.quad.nodes, uq.op) .+
             evaluatePCE(faceR.femL[2, :], uq.op.quad.nodes, uq.op)) ./ dx
        ERan[3, :] .-=
            dt .* (evaluatePCE(faceL.femR[3, :], uq.op.quad.nodes, uq.op) .+
             evaluatePCE(faceR.femL[3, :], uq.op.quad.nodes, uq.op)) ./ dx
        BRan = chaos_ran(cell.B, 2, uq)
        BRan[1, :] .-=
            dt .* (evaluatePCE(faceL.femR[4, :], uq.op.quad.nodes, uq.op) .+
             evaluatePCE(faceR.femL[4, :], uq.op.quad.nodes, uq.op)) ./ dx
        BRan[2, :] .-=
            dt .* (evaluatePCE(faceL.femR[5, :], uq.op.quad.nodes, uq.op) .+
             evaluatePCE(faceR.femL[5, :], uq.op.quad.nodes, uq.op)) ./ dx
        BRan[3, :] .-=
            dt .* (evaluatePCE(faceL.femR[6, :], uq.op.quad.nodes, uq.op) .+
             evaluatePCE(faceR.femL[6, :], uq.op.quad.nodes, uq.op)) ./ dx

        # source -> ϕ
        #@. cell.ϕ += dt * (cell.w[1,:,1] / KS.gas.mi - cell.w[1,:,2] / KS.gas.me) / (KS.gas.lD^2 * KS.gas.rL)

        # source -> U^{n+1}, E^{n+1} and B^{n+1}
        mr = KS.gas.mi / KS.gas.me

        #ERan = get_ran_array(cell.E, 2, uq)
        #BRan = get_ran_array(cell.B, 2, uq)

        xRan = zeros(9, uq.op.quad.Nquad)
        for j in axes(xRan, 2)
            A, b = em_coefficients(
                primRan[:, j, :],
                ERan[:, j],
                BRan[:, j],
                mr,
                KS.gas.lD,
                KS.gas.rL,
                dt,
            )
            xRan[:, j] .= A \ b
        end

        lorenzRan = zeros(3, uq.op.quad.Nquad, 2)
        for j in axes(lorenzRan, 2)
            lorenzRan[1, j, 1] =
                0.5 *
                (xRan[1, j] + ERan[1, j] + (primRan[3, j, 1] + xRan[5, j]) * BRan[3, j] -
                 (primRan[4, j, 1] + xRan[6, j]) * BRan[2, j]) / KS.gas.rL
            lorenzRan[2, j, 1] =
                0.5 *
                (xRan[2, j] + ERan[2, j] + (primRan[4, j, 1] + xRan[6, j]) * BRan[1, j] -
                 (primRan[2, j, 1] + xRan[4, j]) * BRan[3, j]) / KS.gas.rL
            lorenzRan[3, j, 1] =
                0.5 *
                (xRan[3, j] + ERan[3, j] + (primRan[2, j, 1] + xRan[4, j]) * BRan[2, j] -
                 (primRan[3, j, 1] + xRan[5, j]) * BRan[1, j]) / KS.gas.rL
            lorenzRan[1, j, 2] =
                -0.5 *
                (xRan[1, j] + ERan[1, j] + (primRan[3, j, 2] + xRan[8, j]) * BRan[3, j] -
                 (primRan[4, j, 2] + xRan[9, j]) * BRan[2, j]) *
                mr / KS.gas.rL
            lorenzRan[2, j, 2] =
                -0.5 *
                (xRan[2, j] + ERan[2, j] + (primRan[4, j, 2] + xRan[9, j]) * BRan[1, j] -
                 (primRan[2, j, 2] + xRan[7, j]) * BRan[3, j]) *
                mr / KS.gas.rL
            lorenzRan[3, j, 2] =
                -0.5 *
                (xRan[3, j] + ERan[3, j] + (primRan[2, j, 2] + xRan[7, j]) * BRan[2, j] -
                 (primRan[3, j, 2] + xRan[8, j]) * BRan[1, j]) *
                mr / KS.gas.rL
        end

        ERan[1, :] .= xRan[1, :]
        ERan[2, :] .= xRan[2, :]
        ERan[3, :] .= xRan[3, :]

        #--- update conservative flow variables: step 2 ---#
        primRan[2, :, 1] .= xRan[4, :]
        primRan[3, :, 1] .= xRan[5, :]
        primRan[4, :, 1] .= xRan[6, :]
        primRan[2, :, 2] .= xRan[7, :]
        primRan[3, :, 2] .= xRan[8, :]
        primRan[4, :, 2] .= xRan[9, :]

        for j in axes(wRan, 2)
            wRan[:, j, :] .= KitBase.mixture_prim_conserve(primRan[:, j, :], KS.gas.γ)
        end

        cell.w .= ran_chaos(wRan, 2, uq)
        cell.prim .= ran_chaos(primRan, 2, uq)
        cell.E .= ran_chaos(ERan, 2, uq)
        cell.lorenz .= ran_chaos(lorenzRan, 2, uq)
        cell.B .= ran_chaos(BRan, 2, uq)

        #--- update particle distribution function ---#
        # flux -> f^{n+1}
        #@. cell.h0 += (faceL.fh0 - faceR.fh0) / dx
        #@. cell.h1 += (faceL.fh1 - faceR.fh1) / dx
        #@. cell.h2 += (faceL.fh2 - faceR.fh2) / dx
        #@. cell.h3 += (faceL.fh3 - faceR.fh3) / dx

        #h0Ran = get_ran_array(cell.h0, 2, uq)
        #h1Ran = get_ran_array(cell.h1, 2, uq)
        #h2Ran = get_ran_array(cell.h2, 2, uq)
        #h3Ran = get_ran_array(cell.h3, 2, uq)

        h0Ran =
            chaos_ran(cell.h0, 2, uq) .+
            (chaos_ran(faceL.fh0, 2, uq) .- chaos_ran(faceR.fh0, 2, uq)) ./ dx
        h1Ran =
            chaos_ran(cell.h1, 2, uq) .+
            (chaos_ran(faceL.fh1, 2, uq) .- chaos_ran(faceR.fh1, 2, uq)) ./ dx
        h2Ran =
            chaos_ran(cell.h2, 2, uq) .+
            (chaos_ran(faceL.fh2, 2, uq) .- chaos_ran(faceR.fh2, 2, uq)) ./ dx
        h3Ran =
            chaos_ran(cell.h3, 2, uq) .+
            (chaos_ran(faceL.fh3, 2, uq) .- chaos_ran(faceR.fh3, 2, uq)) ./ dx

        # force -> f^{n+1} : step 1
        for j in axes(h0Ran, 2)
            _h0 = @view h0Ran[:, j, :]
            _h1 = @view h1Ran[:, j, :]
            _h2 = @view h2Ran[:, j, :]
            _h3 = @view h3Ran[:, j, :]

            shift_pdf!(_h0, lorenzRan[1, j, :], KS.vs.du[1, :], dt)
            shift_pdf!(_h1, lorenzRan[1, j, :], KS.vs.du[1, :], dt)
            shift_pdf!(_h2, lorenzRan[1, j, :], KS.vs.du[1, :], dt)
            shift_pdf!(_h3, lorenzRan[1, j, :], KS.vs.du[1, :], dt)
        end

        # force -> f^{n+1} : step 2
        for k in axes(h1Ran, 3), j in axes(h1Ran, 2)
            @. h3Ran[:, j, k] +=
                2.0 * dt * lorenzRan[2, j, k] * h1Ran[:, j, k] +
                (dt * lorenzRan[2, j, k])^2 * h0Ran[:, j, k] +
                2.0 * dt * lorenzRan[3, j, k] * h2Ran[:, j, k] +
                (dt * lorenzRan[3, j, k])^2 * h0Ran[:, j, k]
            @. h2Ran[:, j, k] += dt * lorenzRan[3, j, k] * h0Ran[:, j, k]
            @. h1Ran[:, j, k] += dt * lorenzRan[2, j, k] * h0Ran[:, j, k]
        end

        # source -> f^{n+1}
        #tau = get_tau(cell.prim, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1], uq)
        tau = get_tau(primRan, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1])

        # interspecies interaction
        for j in axes(primRan, 2)
            primRan[:, j, :] .= KitBase.aap_hs_prim(
                primRan[:, j, :],
                tau,
                KS.gas.mi,
                KS.gas.ni,
                KS.gas.me,
                KS.gas.ne,
                KS.gas.Kn[1],
            )
        end

        gRan = zeros(KS.vs.nu, uq.op.quad.Nquad, 2)
        for k in axes(gRan, 3)
            for j in axes(gRan, 2)
                gRan[:, j, k] .= KitBase.maxwellian(KS.vs.u[:, k], primRan[:, j, k])
            end
        end

        # BGK term
        for j in axes(h0Ran, 2)
            Mu, Mv, Mw, MuL, MuR = KitBase.mixture_gauss_moments(primRan[:, j, :], KS.gas.K)
            for k in axes(h0Ran, 3)
                @. h0Ran[:, j, k] =
                    (h0Ran[:, j, k] + dt / tau[k] * gRan[:, j, k]) / (1.0 + dt / tau[k])
                @. h1Ran[:, j, k] =
                    (h1Ran[:, j, k] + dt / tau[k] * Mv[1, k] * gRan[:, j, k]) /
                    (1.0 + dt / tau[k])
                @. h2Ran[:, j, k] =
                    (h2Ran[:, j, k] + dt / tau[k] * Mw[1, k] * gRan[:, j, k]) /
                    (1.0 + dt / tau[k])
                @. h3Ran[:, j, k] =
                    (h3Ran[:, j, k] + dt / tau[k] * (Mv[2, k] + Mw[2, k]) * gRan[:, j, k]) /
                    (1.0 + dt / tau[k])
            end
        end

        cell.h0 .= ran_chaos(h0Ran, 2, uq)
        cell.h1 .= ran_chaos(h1Ran, 2, uq)
        cell.h2 .= ran_chaos(h2Ran, 2, uq)
        cell.h3 .= ran_chaos(h3Ran, 2, uq)

        #--- record residuals ---#
        @. RES += (w_old[:, 1, :] - cell.w[:, 1, :])^2
        @. AVG += abs(cell.w[:, 1, :])
        #=
        #-- filter ---#
        λ = 0.00001
        for k in 1:2, i in 1:5
        filter!(cell.w[i,:,k], λ)
        filter!(cell.prim[i,:,k], λ)
        end
        for k in 1:2, i in axes(cell.h0, 1)
        filter!(cell.h0[i,:,k], λ)
        filter!(cell.h1[i,:,k], λ)
        filter!(cell.h2[i,:,k], λ)
        filter!(cell.h3[i,:,k], λ)
        end
        for k in 1:2, i in 1:3
        filter!(cell.lorenz[i,:,k], λ)
        end
        for i in 1:3
        filter!(cell.E[i,:], λ)
        filter!(cell.B[i,:], λ)
        end
        filter!(cell.ϕ, λ)
        filter!(cell.ψ, λ)
        =#

    elseif uq.method == "collocation"

        #--- update conservative flow variables: step 1 ---#
        # w^n
        w_old = deepcopy(cell.w)
        prim_old = deepcopy(cell.prim)

        # flux -> w^{n+1}
        @. cell.w += (faceL.fw - faceR.fw) / dx
        for j in axes(cell.prim, 2)
            cell.prim[:, j, :] .= mixture_conserve_prim(cell.w[:, j, :], KS.gas.γ)
        end

        # temperature protection
        if min(minimum(cell.prim[5, :, 1]), minimum(cell.prim[5, :, 2])) < 0
            @warn "negative temperature update"
            cell.w .= w_old
            cell.prim .= prim_old
        end

        #=
        # source -> w^{n+1}
        # DifferentialEquations.jl
        tau = get_tau(cell.prim, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1])
        for j in axes(wRan, 2)
        prob = ODEProblem( mixture_source,
            vcat(cell.w[1:5,j,1], cell.w[1:5,j,2]),
            dt,
            (tau[1], tau[2], KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1], KS.gas.γ) )
        sol = solve(prob, Rosenbrock23())

        cell.w[1:5,j,1] .= sol[end][1:5]
        cell.w[1:5,j,2] .= sol[end][6:10]
        for k=1:2
        cell.prim[:,j,k] .= KitBase.conserve_prim(cell.w[:,j,k], KS.gas.γ)
        end
        end
        =#
        #=
        # explicit
        tau = get_tau(cell.prim, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1])
        mprim = get_mixprim(cell.prim, tau, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1])
        mw = get_conserved(mprim, KS.gas.γ)
        for k=1:2
        cell.w[:,:,k] .+= (mw[:,:,k] .- cell.w[:,:,k]) .* dt ./ tau[k]
        end
        cell.prim .= get_primitive(cell.w, KS.gas.γ)
        =#

        #--- update electromagnetic variables ---#
        # flux -> E^{n+1} & B^{n+1}
        @. cell.E[1, :] -= dt * (faceL.femR[1, :] + faceR.femL[1, :]) / dx
        @. cell.E[2, :] -= dt * (faceL.femR[2, :] + faceR.femL[2, :]) / dx
        @. cell.E[3, :] -= dt * (faceL.femR[3, :] + faceR.femL[3, :]) / dx
        @. cell.B[1, :] -= dt * (faceL.femR[4, :] + faceR.femL[4, :]) / dx
        @. cell.B[2, :] -= dt * (faceL.femR[5, :] + faceR.femL[5, :]) / dx
        @. cell.B[3, :] -= dt * (faceL.femR[6, :] + faceR.femL[6, :]) / dx
        @. cell.ϕ -= dt * (faceL.femR[7, :] + faceR.femL[7, :]) / dx
        @. cell.ψ -= dt * (faceL.femR[8, :] + faceR.femL[8, :]) / dx

        # source -> ϕ
        #@. cell.ϕ += dt * (cell.w[1,:,1] / KS.gas.mi - cell.w[1,:,2] / KS.gas.me) / (KS.gas.lD^2 * KS.gas.rL)

        # source -> U^{n+1}, E^{n+1} and B^{n+1}
        mr = KS.gas.mi / KS.gas.me

        x = zeros(9, uq.op.quad.Nquad)
        for j in axes(x, 2)
            A, b = em_coefficients(
                cell.prim[:, j, :],
                cell.E[:, j],
                cell.B[:, j],
                mr,
                KS.gas.lD,
                KS.gas.rL,
                dt,
            )
            x[:, j] .= A \ b
        end

        #--- calculate lorenz force ---#
        for j in axes(cell.lorenz, 2)
            cell.lorenz[1, j, 1] =
                0.5 *
                (x[1, j] + cell.E[1, j] + (cell.prim[3, j, 1] + x[5, j]) * cell.B[3, j] -
                 (cell.prim[4, j, 1] + x[6, j]) * cell.B[2, j]) / KS.gas.rL
            cell.lorenz[2, j, 1] =
                0.5 *
                (x[2, j] + cell.E[2, j] + (cell.prim[4, j, 1] + x[6, j]) * cell.B[1, j] -
                 (cell.prim[2, j, 1] + x[4, j]) * cell.B[3, j]) / KS.gas.rL
            cell.lorenz[3, j, 1] =
                0.5 *
                (x[3, j] + cell.E[3, j] + (cell.prim[2, j, 1] + x[4, j]) * cell.B[2, j] -
                 (cell.prim[3, j, 1] + x[5, j]) * cell.B[1, j]) / KS.gas.rL
            cell.lorenz[1, j, 2] =
                -0.5 *
                (x[1, j] + cell.E[1, j] + (cell.prim[3, j, 2] + x[8, j]) * cell.B[3, j] -
                 (cell.prim[4, j, 2] + x[9, j]) * cell.B[2, j]) *
                mr / KS.gas.rL
            cell.lorenz[2, j, 2] =
                -0.5 *
                (x[2, j] + cell.E[2, j] + (cell.prim[4, j, 2] + x[9, j]) * cell.B[1, j] -
                 (cell.prim[2, j, 2] + x[7, j]) * cell.B[3, j]) *
                mr / KS.gas.rL
            cell.lorenz[3, j, 2] =
                -0.5 *
                (x[3, j] + cell.E[3, j] + (cell.prim[2, j, 2] + x[7, j]) * cell.B[2, j] -
                 (cell.prim[3, j, 2] + x[8, j]) * cell.B[1, j]) *
                mr / KS.gas.rL
        end

        cell.E[1, :] .= x[1, :]
        cell.E[2, :] .= x[2, :]
        cell.E[3, :] .= x[3, :]

        #--- update conservative flow variables: step 2 ---#
        cell.prim[2, :, 1] .= x[4, :]
        cell.prim[3, :, 1] .= x[5, :]
        cell.prim[4, :, 1] .= x[6, :]
        cell.prim[2, :, 2] .= x[7, :]
        cell.prim[3, :, 2] .= x[8, :]
        cell.prim[4, :, 2] .= x[9, :]

        for j in axes(cell.w, 2)
            cell.w[:, j, :] .= KitBase.mixture_prim_conserve(cell.prim[:, j, :], KS.gas.γ)
        end

        #--- update particle distribution function ---#
        # flux -> f^{n+1}
        @. cell.h0 += (faceL.fh0 - faceR.fh0) / dx
        @. cell.h1 += (faceL.fh1 - faceR.fh1) / dx
        @. cell.h2 += (faceL.fh2 - faceR.fh2) / dx
        @. cell.h3 += (faceL.fh3 - faceR.fh3) / dx

        # force -> f^{n+1} : step 1
        for k in axes(cell.h0, 3)
            for j in axes(cell.h0, 2)
                _h0 = @view cell.h0[:, j, k]
                _h1 = @view cell.h1[:, j, k]
                _h2 = @view cell.h2[:, j, k]
                _h3 = @view cell.h3[:, j, k]

                shift_pdf!(_h0, cell.lorenz[1, j, k], KS.vs.du[1, k], dt)
                shift_pdf!(_h1, cell.lorenz[1, j, k], KS.vs.du[1, k], dt)
                shift_pdf!(_h2, cell.lorenz[1, j, k], KS.vs.du[1, k], dt)
                shift_pdf!(_h3, cell.lorenz[1, j, k], KS.vs.du[1, k], dt)
            end
        end

        # force -> f^{n+1} : step 2
        for k in axes(cell.h1, 3), j in axes(cell.h1, 2)
            @. cell.h3[:, j, k] +=
                2.0 * dt * cell.lorenz[2, j, k] * cell.h1[:, j, k] +
                (dt * cell.lorenz[2, j, k])^2 * cell.h0[:, j, k] +
                2.0 * dt * cell.lorenz[3, j, k] * cell.h2[:, j, k] +
                (dt * cell.lorenz[3, j, k])^2 * cell.h0[:, j, k]
            @. cell.h2[:, j, k] += dt * cell.lorenz[3, j, k] * cell.h0[:, j, k]
            @. cell.h1[:, j, k] += dt * cell.lorenz[2, j, k] * cell.h0[:, j, k]
        end

        # source -> f^{n+1}
        tau = uq_aap_hs_collision_time(
            cell.prim,
            KS.gas.mi,
            KS.gas.ni,
            KS.gas.me,
            KS.gas.ne,
            KS.gas.Kn[1],
            uq,
        )

        # interspecies interaction
        prim = deepcopy(cell.prim)
        #for j in axes(prim, 2)
        #    prim[:, j, :] .= KitBase.aap_hs_prim(
        #        cell.prim[:, j, :],
        #        tau,
        #        KS.gas.mi,
        #        KS.gas.ni,
        #        KS.gas.me,
        #        KS.gas.ne,
        #        KS.gas.Kn[1],
        #    )
        #end

        g = zeros(KS.vs.nu, uq.op.quad.Nquad, 2)
        for j in axes(g, 2)
            g[:, j, :] .= mixture_maxwellian(KS.vs.u, prim[:, j, :])
        end

        # BGK term
        for j in axes(cell.h0, 2)
            Mu, Mv, Mw, MuL, MuR = mixture_gauss_moments(prim[:, j, :], KS.gas.K)
            for k in axes(cell.h0, 3)
                @. cell.h0[:, j, k] =
                    (cell.h0[:, j, k] + dt / tau[k] * g[:, j, k]) / (1.0 + dt / tau[k])
                @. cell.h1[:, j, k] =
                    (cell.h1[:, j, k] + dt / tau[k] * Mv[1, k] * g[:, j, k]) /
                    (1.0 + dt / tau[k])
                @. cell.h2[:, j, k] =
                    (cell.h2[:, j, k] + dt / tau[k] * Mw[1, k] * g[:, j, k]) /
                    (1.0 + dt / tau[k])
                @. cell.h3[:, j, k] =
                    (cell.h3[:, j, k] + dt / tau[k] * (Mv[2, k] + Mw[2, k]) * g[:, j, k]) /
                    (1.0 + dt / tau[k])
            end
        end

        #--- record residuals ---#
        @. RES += (w_old[:, 1, :] - cell.w[:, 1, :])^2
        @. AVG += abs(cell.w[:, 1, :])
    end
end

"""
$(SIGNATURES)

1D3F2V mixture
"""
function step!(
    KS,
    uq::AbstractUQ,
    faceL,
    cell::ControlVolume1D3F,
    faceR,
    p,
    coll=:bgk,
    isMHD=true,
)
    dt, dx, RES, AVG = p

    if uq.method == "galerkin"
        @assert size(cell.w, 2) == uq.nr + 1

        #--- update conservative flow variables: step 1 ---#
        # w^n
        w_old = deepcopy(cell.w)
        prim_old = deepcopy(cell.prim)

        # flux -> w^{n+1}
        @. cell.w += (faceL.fw - faceR.fw) / dx
        cell.prim .= get_primitive(cell.w, KS.gas.γ, uq)

        # locate variables on random quadrature points
        wRan = chaos_ran(cell.w, 2, uq)
        primRan = chaos_ran(cell.prim, 2, uq)

        # temperature protection
        if min(minimum(primRan[5, :, 1]), minimum(primRan[5, :, 2])) < 0
            @warn "negative temperature update"
            wRan = chaos_ran(w_old, 2, uq)
            primRan = chaos_ran(prim_old, 2, uq)
        end

        if isMHD == false
            tauRan = uq_aap_hs_collision_time(
                primRan,
                KS.gas.mi,
                KS.gas.ni,
                KS.gas.me,
                KS.gas.ne,
                KS.gas.Kn[1],
                uq,
            )
            mprimRan = uq_aap_hs_prim(
                primRan,
                tauRan,
                KS.gas.mi,
                KS.gas.ni,
                KS.gas.me,
                KS.gas.ne,
                KS.gas.Kn[1],
                uq,
            )
            mwRan = uq_prim_conserve(mprimRan, KS.gas.γ, uq)
            for k in axes(wRan, 3)
                @. wRan[:, :, k] += (mwRan[:, :, k] - wRan[:, :, k]) * dt / tauRan[k]
            end
            primRan .= uq_conserve_prim(wRan, KS.gas.γ, uq)
        end

        #--- update electromagnetic variables ---#
        # flux -> E^{n+1} & B^{n+1}
        @. cell.E[1, :] -= dt * (faceL.femR[1, :] + faceR.femL[1, :]) / dx
        @. cell.E[2, :] -= dt * (faceL.femR[2, :] + faceR.femL[2, :]) / dx
        @. cell.E[3, :] -= dt * (faceL.femR[3, :] + faceR.femL[3, :]) / dx
        @. cell.B[1, :] -= dt * (faceL.femR[4, :] + faceR.femL[4, :]) / dx
        @. cell.B[2, :] -= dt * (faceL.femR[5, :] + faceR.femL[5, :]) / dx
        @. cell.B[3, :] -= dt * (faceL.femR[6, :] + faceR.femL[6, :]) / dx
        @. cell.ϕ -= dt * (faceL.femR[7, :] + faceR.femL[7, :]) / dx
        @. cell.ψ -= dt * (faceL.femR[8, :] + faceR.femL[8, :]) / dx

        ERan = chaos_ran(cell.E, 2, uq)
        BRan = chaos_ran(cell.B, 2, uq)

        # source -> ϕ
        #@. cell.ϕ += dt * (cell.w[1,:,1] / KS.gas.mi - cell.w[1,:,2] / KS.gas.me) / (KS.gas.lD^2 * KS.gas.rL)

        # source -> U^{n+1}, E^{n+1} and B^{n+1}
        mr = KS.gas.mi / KS.gas.me

        xRan = zeros(9, uq.op.quad.Nquad)
        for j in axes(xRan, 2)
            A, b = em_coefficients(
                primRan[:, j, :],
                ERan[:, j],
                BRan[:, j],
                mr,
                KS.gas.lD,
                KS.gas.rL,
                dt,
            )
            xRan[:, j] .= A \ b
        end

        lorenzRan = zeros(3, uq.op.quad.Nquad, 2)
        for j in axes(lorenzRan, 2)
            lorenzRan[1, j, 1] =
                0.5 *
                (xRan[1, j] + ERan[1, j] + (primRan[3, j, 1] + xRan[5, j]) * BRan[3, j] -
                 (primRan[4, j, 1] + xRan[6, j]) * BRan[2, j]) / KS.gas.rL
            lorenzRan[2, j, 1] =
                0.5 *
                (xRan[2, j] + ERan[2, j] + (primRan[4, j, 1] + xRan[6, j]) * BRan[1, j] -
                 (primRan[2, j, 1] + xRan[4, j]) * BRan[3, j]) / KS.gas.rL
            lorenzRan[3, j, 1] =
                0.5 *
                (xRan[3, j] + ERan[3, j] + (primRan[2, j, 1] + xRan[4, j]) * BRan[2, j] -
                 (primRan[3, j, 1] + xRan[5, j]) * BRan[1, j]) / KS.gas.rL
            lorenzRan[1, j, 2] =
                -0.5 *
                (xRan[1, j] + ERan[1, j] + (primRan[3, j, 2] + xRan[8, j]) * BRan[3, j] -
                 (primRan[4, j, 2] + xRan[9, j]) * BRan[2, j]) *
                mr / KS.gas.rL
            lorenzRan[2, j, 2] =
                -0.5 *
                (xRan[2, j] + ERan[2, j] + (primRan[4, j, 2] + xRan[9, j]) * BRan[1, j] -
                 (primRan[2, j, 2] + xRan[7, j]) * BRan[3, j]) *
                mr / KS.gas.rL
            lorenzRan[3, j, 2] =
                -0.5 *
                (xRan[3, j] + ERan[3, j] + (primRan[2, j, 2] + xRan[7, j]) * BRan[2, j] -
                 (primRan[3, j, 2] + xRan[8, j]) * BRan[1, j]) *
                mr / KS.gas.rL
        end

        ERan[1, :] .= xRan[1, :]
        ERan[2, :] .= xRan[2, :]
        ERan[3, :] .= xRan[3, :]

        #--- update conservative flow variables: step 2 ---#
        primRan[2, :, 1] .= xRan[4, :]
        primRan[3, :, 1] .= xRan[5, :]
        primRan[4, :, 1] .= xRan[6, :]
        primRan[2, :, 2] .= xRan[7, :]
        primRan[3, :, 2] .= xRan[8, :]
        primRan[4, :, 2] .= xRan[9, :]

        for j in axes(wRan, 2)
            wRan[:, j, :] .= KitBase.mixture_prim_conserve(primRan[:, j, :], KS.gas.γ)
        end

        cell.w .= ran_chaos(wRan, 2, uq)
        cell.prim .= ran_chaos(primRan, 2, uq)
        cell.E .= ran_chaos(ERan, 2, uq)
        cell.lorenz .= ran_chaos(lorenzRan, 2, uq)
        cell.B .= ran_chaos(BRan, 2, uq)

        #--- update particle distribution function ---#
        # flux -> f^{n+1}
        @. cell.h0 += (faceL.fh0 - faceR.fh0) / dx
        @. cell.h1 += (faceL.fh1 - faceR.fh1) / dx
        @. cell.h2 += (faceL.fh2 - faceR.fh2) / dx

        h0Ran = chaos_ran(cell.h0, 3, uq)
        h1Ran = chaos_ran(cell.h1, 3, uq)
        h2Ran = chaos_ran(cell.h2, 3, uq)

        # force -> f^{n+1} : step 1
        for k in axes(h0Ran, 4)
            for j in axes(h0Ran, 3)
                for i in axes(h0Ran, 2)
                    _h0 = @view h0Ran[:, i, j, k]
                    _h1 = @view h1Ran[:, i, j, k]
                    _h2 = @view h2Ran[:, i, j, k]

                    shift_pdf!(_h0, cell.lorenz[1, j, k], KS.vs.du[1, i, k], dt)
                    shift_pdf!(_h1, cell.lorenz[1, j, k], KS.vs.du[1, i, k], dt)
                    shift_pdf!(_h2, cell.lorenz[1, j, k], KS.vs.du[1, i, k], dt)
                end
            end
        end

        for k in axes(h0Ran, 4)
            for j in axes(h0Ran, 3)
                for i in axes(h0Ran, 1)
                    _h0 = @view h0Ran[i, :, j, k]
                    _h1 = @view h1Ran[i, :, j, k]
                    _h2 = @view h2Ran[i, :, j, k]

                    shift_pdf!(_h0, lorenzRan[2, j, k], KS.vs.dv[i, 1, k], dt)
                    shift_pdf!(_h1, lorenzRan[2, j, k], KS.vs.dv[i, 1, k], dt)
                    shift_pdf!(_h2, lorenzRan[2, j, k], KS.vs.dv[i, 1, k], dt)
                end
            end
        end

        # force -> f^{n+1} : step 2
        for k in axes(h1Ran, 4), j in axes(h1Ran, 3)
            @. h2Ran[:, :, j, k] +=
                2.0 * dt * lorenzRan[3, j, k] * h1Ran[:, :, j, k] +
                (dt * lorenzRan[3, j, k])^2 * h0Ran[:, :, j, k]
            @. h1Ran[:, :, j, k] += dt * lorenzRan[3, j, k] * h0Ran[:, :, j, k]
        end

        # source -> f^{n+1}
        tau = uq_aap_hs_collision_time(
            primRan,
            KS.gas.mi,
            KS.gas.ni,
            KS.gas.me,
            KS.gas.ne,
            KS.gas.Kn[1],
            uq,
        )

        if isMHD == false
            # interspecies interaction
            for j in axes(primRan, 2)
                primRan[:, j, :] .= KitBase.aap_hs_prim(
                    primRan[:, j, :],
                    tau,
                    KS.gas.mi,
                    KS.gas.ni,
                    KS.gas.me,
                    KS.gas.ne,
                    KS.gas.Kn[1],
                )
            end
        end

        H0Ran = zeros(KS.vs.nu, KS.vs.nv, uq.op.quad.Nquad, 2)
        H1Ran = similar(H0Ran)
        H2Ran = similar(H0Ran)
        for k in axes(H0Ran, 4), j in axes(H0Ran, 3)
            H0Ran[:, :, j, k] .=
                maxwellian(KS.vs.u[:, :, k], KS.vs.v[:, :, k], primRan[:, j, k])
            @. H1Ran[:, :, j, k] = H0Ran[:, :, j, k] * primRan[4, j, k]
            @. H2Ran[:, :, j, k] =
                H0Ran[:, :, j, k] * (primRan[4, j, k]^2 + 1.0 / (2.0 * primRan[5, j, k]))
        end

        # BGK term
        for k in axes(h0Ran, 4)
            for j in axes(h0Ran, 3)
                @. h0Ran[:, :, j, k] =
                    (h0Ran[:, :, j, k] + dt / tau[k] * H0Ran[:, :, j, k]) /
                    (1.0 + dt / tau[k])
                @. h1Ran[:, :, j, k] =
                    (h1Ran[:, :, j, k] + dt / tau[k] * H1Ran[:, :, j, k]) /
                    (1.0 + dt / tau[k])
                @. h2Ran[:, j, :, k] =
                    (h2Ran[:, :, j, k] + dt / tau[k] * H2Ran[:, :, j, k]) /
                    (1.0 + dt / tau[k])
            end
        end

        cell.h0 .= ran_chaos(h0Ran, 3, uq)
        cell.h1 .= ran_chaos(h1Ran, 3, uq)
        cell.h2 .= ran_chaos(h2Ran, 3, uq)

        #--- record residuals ---#
        @. RES += (w_old[:, 1, :] - cell.w[:, 1, :])^2
        @. AVG += abs(cell.w[:, 1, :])

    elseif uq.method == "collocation"
        w_old = deepcopy(cell.w)
        prim_old = deepcopy(cell.prim)

        # flux -> w^{n+1}
        @. cell.w += (faceL.fw - faceR.fw) / dx
        cell.prim = uq_conserve_prim(cell.w, KS.gas.γ, uq)

        # temperature protection
        if min(minimum(cell.prim[5, :, 1]), minimum(cell.prim[5, :, 2])) < 0
            @warn "negative temperature update"
            cell.w .= w_old
            cell.prim .= prim_old
        end

        if isMHD == false
            # explicit
            tau = uq_aap_hs_collision_time(
                cell.prim,
                KS.gas.mi,
                KS.gas.ni,
                KS.gas.me,
                KS.gas.ne,
                KS.gas.Kn[1],
                uq,
            )
            mprim = uq_aap_hs_prim(
                cell.prim,
                tau,
                KS.gas.mi,
                KS.gas.ni,
                KS.gas.me,
                KS.gas.ne,
                KS.gas.Kn[1],
                uq,
            )
            mw = uq_prim_conserve(mprim, KS.gas.γ, uq)
            for k in 1:2
                @. cell.w[:, :, k] += (mw[:, :, k] - cell.w[:, :, k]) * dt / tau[k]
            end
            cell.prim .= uq_conserve_prim(cell.w, KS.gas.γ, uq)
        end

        #--- update electromagnetic variables ---#
        # flux -> E^{n+1} & B^{n+1}
        @. cell.E[1, :] -= dt * (faceL.femR[1, :] + faceR.femL[1, :]) / dx
        @. cell.E[2, :] -= dt * (faceL.femR[2, :] + faceR.femL[2, :]) / dx
        @. cell.E[3, :] -= dt * (faceL.femR[3, :] + faceR.femL[3, :]) / dx
        @. cell.B[1, :] -= dt * (faceL.femR[4, :] + faceR.femL[4, :]) / dx
        @. cell.B[2, :] -= dt * (faceL.femR[5, :] + faceR.femL[5, :]) / dx
        @. cell.B[3, :] -= dt * (faceL.femR[6, :] + faceR.femL[6, :]) / dx
        @. cell.ϕ -= dt * (faceL.femR[7, :] + faceR.femL[7, :]) / dx
        @. cell.ψ -= dt * (faceL.femR[8, :] + faceR.femL[8, :]) / dx

        # source -> ϕ
        #@. cell.ϕ += dt * (cell.w[1,:,1] / KS.gas.mi - cell.w[1,:,2] / KS.gas.me) / (KS.gas.lD^2 * KS.gas.rL)

        # source -> U^{n+1}, E^{n+1} and B^{n+1}
        mr = KS.gas.mi / KS.gas.me

        x = zeros(9, uq.op.quad.Nquad)
        for j in axes(x, 2)
            A, b = em_coefficients(
                cell.prim[:, j, :],
                cell.E[:, j],
                cell.B[:, j],
                mr,
                KS.gas.lD,
                KS.gas.rL,
                dt,
            )
            x[:, j] .= A \ b
        end

        #--- calculate lorenz force ---#
        for j in axes(cell.lorenz, 2)
            cell.lorenz[1, j, 1] =
                0.5 *
                (x[1, j] + cell.E[1, j] + (cell.prim[3, j, 1] + x[5, j]) * cell.B[3, j] -
                 (cell.prim[4, j, 1] + x[6, j]) * cell.B[2, j]) / KS.gas.rL
            cell.lorenz[2, j, 1] =
                0.5 *
                (x[2, j] + cell.E[2, j] + (cell.prim[4, j, 1] + x[6, j]) * cell.B[1, j] -
                 (cell.prim[2, j, 1] + x[4, j]) * cell.B[3, j]) / KS.gas.rL
            cell.lorenz[3, j, 1] =
                0.5 *
                (x[3, j] + cell.E[3, j] + (cell.prim[2, j, 1] + x[4, j]) * cell.B[2, j] -
                 (cell.prim[3, j, 1] + x[5, j]) * cell.B[1, j]) / KS.gas.rL
            cell.lorenz[1, j, 2] =
                -0.5 *
                (x[1, j] + cell.E[1, j] + (cell.prim[3, j, 2] + x[8, j]) * cell.B[3, j] -
                 (cell.prim[4, j, 2] + x[9, j]) * cell.B[2, j]) *
                mr / KS.gas.rL
            cell.lorenz[2, j, 2] =
                -0.5 *
                (x[2, j] + cell.E[2, j] + (cell.prim[4, j, 2] + x[9, j]) * cell.B[1, j] -
                 (cell.prim[2, j, 2] + x[7, j]) * cell.B[3, j]) *
                mr / KS.gas.rL
            cell.lorenz[3, j, 2] =
                -0.5 *
                (x[3, j] + cell.E[3, j] + (cell.prim[2, j, 2] + x[7, j]) * cell.B[2, j] -
                 (cell.prim[3, j, 2] + x[8, j]) * cell.B[1, j]) *
                mr / KS.gas.rL
        end

        cell.E[1, :] .= x[1, :]
        cell.E[2, :] .= x[2, :]
        cell.E[3, :] .= x[3, :]

        #--- update conservative flow variables: step 2 ---#
        cell.prim[2, :, 1] .= x[4, :]
        cell.prim[3, :, 1] .= x[5, :]
        cell.prim[4, :, 1] .= x[6, :]
        cell.prim[2, :, 2] .= x[7, :]
        cell.prim[3, :, 2] .= x[8, :]
        cell.prim[4, :, 2] .= x[9, :]

        for j in axes(cell.w, 2)
            cell.w[:, j, :] .= KitBase.mixture_prim_conserve(cell.prim[:, j, :], KS.gas.γ)
        end

        #--- update particle distribution function ---#
        # flux -> f^{n+1}
        @. cell.h0 += (faceL.fh0 - faceR.fh0) / dx
        @. cell.h1 += (faceL.fh1 - faceR.fh1) / dx
        @. cell.h2 += (faceL.fh2 - faceR.fh2) / dx

        # force -> f^{n+1} : step 1
        for k in axes(cell.h0, 4)
            for j in axes(cell.h0, 3)
                for i in axes(cell.h0, 2)
                    _h0 = @view cell.h0[:, i, j, k]
                    _h1 = @view cell.h1[:, i, j, k]
                    _h2 = @view cell.h2[:, i, j, k]

                    shift_pdf!(_h0, cell.lorenz[1, j, k], KS.vs.du[1, i, k], dt)
                    shift_pdf!(_h1, cell.lorenz[1, j, k], KS.vs.du[1, i, k], dt)
                    shift_pdf!(_h2, cell.lorenz[1, j, k], KS.vs.du[1, i, k], dt)
                end
            end
        end

        for k in axes(cell.h0, 4)
            for j in axes(cell.h0, 3)
                for i in axes(cell.h0, 1)
                    _h0 = @view cell.h0[i, :, j, k]
                    _h1 = @view cell.h1[i, :, j, k]
                    _h2 = @view cell.h2[i, :, j, k]

                    shift_pdf!(_h0, cell.lorenz[2, j, k], KS.vs.dv[i, 1, k], dt)
                    shift_pdf!(_h1, cell.lorenz[2, j, k], KS.vs.dv[i, 1, k], dt)
                    shift_pdf!(_h2, cell.lorenz[2, j, k], KS.vs.dv[i, 1, k], dt)
                end
            end
        end

        # force -> f^{n+1} : step 2
        for k in axes(cell.h1, 4), j in axes(cell.h1, 3)
            @. cell.h2[:, :, j, k] +=
                2.0 * dt * cell.lorenz[3, j, k] * cell.h1[:, :, j, k] +
                (dt * cell.lorenz[3, j, k])^2 * cell.h0[:, :, j, k]
            @. cell.h1[:, :, j, k] += dt * cell.lorenz[3, j, k] * cell.h0[:, :, j, k]
        end

        # source -> f^{n+1}
        tau = uq_aap_hs_collision_time(
            cell.prim,
            KS.gas.mi,
            KS.gas.ni,
            KS.gas.me,
            KS.gas.ne,
            KS.gas.Kn[1],
            uq,
        )

        # interspecies interaction
        prim = deepcopy(cell.prim)
        if isMHD == false
            for j in axes(prim, 2)
                prim[:, j, :] .= KitBase.aap_hs_prim(
                    cell.prim[:, j, :],
                    tau,
                    KS.gas.mi,
                    KS.gas.ni,
                    KS.gas.me,
                    KS.gas.ne,
                    KS.gas.Kn[1],
                )
            end
        end

        H0 = zeros(KS.vs.nu, KS.vs.nv, uq.op.quad.Nquad, 2)
        H1 = similar(H0)
        H2 = similar(H0)
        for k in axes(H0, 4), j in axes(H0, 3)
            H0[:, :, j, k] .= maxwellian(KS.vs.u[:, :, k], KS.vs.v[:, :, k], prim[:, j, k])
            @. H1[:, :, j, k] = H0[:, :, j, k] * prim[4, j, k]
            @. H2[:, :, j, k] =
                H0[:, :, j, k] * (prim[4, j, k]^2 + 1.0 / (2.0 * prim[5, j, k]))
        end

        # BGK term
        for k in axes(cell.h0, 4), j in axes(cell.h0, 3)
            @. cell.h0[:, :, j, k] =
                (cell.h0[:, :, j, k] + dt / tau[k] * H0[:, :, j, k]) / (1.0 + dt / tau[k])
            @. cell.h1[:, :, j, k] =
                (cell.h1[:, :, j, k] + dt / tau[k] * H1[:, :, j, k]) / (1.0 + dt / tau[k])
            @. cell.h2[:, :, j, k] =
                (cell.h2[:, :, j, k] + dt / tau[k] * H2[:, :, j, k]) / (1.0 + dt / tau[k])
        end

        #--- record residuals ---#
        @. RES += (w_old[:, 1, :] - cell.w[:, 1, :])^2
        @. AVG += abs(cell.w[:, 1, :])
    end
end

"""
$(SIGNATURES)

2D2F2V
"""
function step!(
    KS,
    uq::AbstractUQ,
    sol::Solution2F{T1,T2,T3,T4,2},
    flux,
    dt,
    residual,
) where {T1,T2,T3,T4}
    sumRes = zeros(axes(KS.ib.wL, 1))
    sumAvg = zeros(axes(KS.ib.wL, 1))

    @threads for j in 1:KS.ps.ny
        for i in 1:KS.ps.nx
            step!(
                KS,
                uq,
                sol.w[i, j],
                sol.prim[i, j],
                sol.h[i, j],
                sol.b[i, j],
                flux.fw1[i, j],
                flux.fh1[i, j],
                flux.fb1[i, j],
                flux.fw1[i+1, j],
                flux.fh1[i+1, j],
                flux.fb1[i+1, j],
                flux.fw2[i, j],
                flux.fh2[i, j],
                flux.fb2[i, j],
                flux.fw2[i, j+1],
                flux.fh2[i, j+1],
                flux.fb2[i, j+1],
                dt,
                KS.ps.dx[i, j] * KS.ps.dy[i, j],
                sumRes,
                sumAvg,
            )
        end
    end

    @. residual = sqrt(sumRes * KS.ps.nx * KS.ps.ny) / (sumAvg + 1.e-7)
end

"""
$(SIGNATURES)

2D2F2V
"""
function step!(
    KS::AbstractSolverSet,
    uq::AbstractUQ,
    w::AM,
    prim::AM,
    h::AA{T,3},
    b::AA{T,3},
    fwL::AM,
    fhL::AA{T,3},
    fbL::AA{T,3},
    fwR::AM,
    fhR::AA{T,3},
    fbR::AA{T,3},
    fwU::AM,
    fhU::AA{T,3},
    fbU::AA{T,3},
    fwD::AM,
    fhD::AA{T,3},
    fbD::AA{T,3},
    dt,
    area,
    sumRes::AV,
    sumAvg::AV,
) where {T}
    w_old = deepcopy(w)

    @. w += (fwL - fwR + fwD - fwU) / area
    prim .= uq_conserve_prim(w, KS.gas.γ, uq)

    τ = uq_vhs_collision_time(prim, KS.gas.μᵣ, KS.gas.ω, uq)
    H = uq_maxwellian(KS.vs.u, KS.vs.v, prim, uq)
    B = similar(H)
    for k in axes(H, 3)
        B[:, :, k] .= H[:, :, k] .* KS.gas.K ./ (2.0 * prim[end, k])
    end

    for k in axes(h, 3)
        @. h[:, :, k] =
            (h[:, :, k] +
             (fhL[:, :, k] - fhR[:, :, k] + fhD[:, :, k] - fhU[:, :, k]) / area +
             dt / τ[k] * H[:, :, k]) / (1.0 + dt / τ[k])
        @. b[:, :, k] =
            (b[:, :, k] +
             (fbL[:, :, k] - fbR[:, :, k] + fbD[:, :, k] - fbU[:, :, k]) / area +
             dt / τ[k] * B[:, :, k]) / (1.0 + dt / τ[k])
    end

    for i in 1:4
        sumRes[i] += sum((w[i, :] .- w_old[i, :]) .^ 2)
        sumAvg[i] += sum(abs.(w[i, :]))
    end
end
