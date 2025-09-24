"""
$(SIGNATURES)

Positivity preserving limiter for collocation (nodal) representation

_R. Vandenhoeck and A. Lani. Implicit high-order flux reconstruction solver for high-speed compressible flows. Computer Physics Communications 242: 1-24, 2019._

## Arguments
- `u`: solution values at quadrature points
- `umean`: mean value of the solution
- `t`: limiter strength in [0, 1]
"""
function collo_positive_limiter!(u::AV, umean, t)
    for i in axes(u, 1)
        u[i] = t * u[i] + (1 - t) * umean
    end

    return nothing
end


"""
$(SIGNATURES)

Determine limiter strength for positivity preservation

_R. Vandenhoeck and A. Lani. Implicit high-order flux reconstruction solver for high-speed compressible flows. Computer Physics Communications 242: 1-24, 2019._
"""
function collo_positive_parameter(u::AV)
    umean = mean(u)
    umin = minimum(u)
    ϵ = min(1e-13, umean)

    t = min((umean - ϵ) / (umean - umin), 1.0)
    @assert 0 <= t <= 1 "incorrect range of limiter parameter"

    return umean, t
end


"""
$(SIGNATURES)

Positivity preserving limiter
"""
function positive_limiter!(u::AV, uq)
    if size(u, 1) == uq.nm + 1 && uq.nm + 1 != uq.nq
        uquad = chaos_ran(u, uq)
        umean, t = collo_positive_parameter(uquad)
        collo_positive_limiter!(uquad, umean, t)
        u .= ran_chaos(uquad, uq)
    elseif size(u, 1) == uq.nq
        umean, t = collo_positive_parameter(u)
        collo_positive_limiter!(u, umean, t)
    end

    return nothing
end

"""
$(SIGNATURES)

Positivity preserving limiter

## Arguments
- `u`: stochastic Galerkin/collocation solution coefficients
- `uq`: uncertainty quantification struct
"""
function positive_limiter!(u::AM, uq)
    if size(u, 2) == uq.nm + 1 && uq.nm + 1 != uq.nq
        uquad = chaos_ran(u, 2, uq)
        ps = begin
            _ps = [collo_positive_parameter(uquad[i, :]) for i in axes(uquad, 1)]
            permutedims(mapreduce(collect, hcat, _ps))
        end
        umeans = ps[:, 1]
        t = minimum(ps[:, 2])
        for i in axes(uquad, 1)
            _u = view(uquad, i, :)
            collo_positive_limiter!(_u, umeans[i], t)
        end
        u .= ran_chaos(uquad, 2, uq)
    elseif size(u, 2) == uq.nq
        ps = begin
            _ps = [collo_positive_parameter(u[i, :]) for i in axes(u, 1)]
            permutedims(mapreduce(collect, hcat, _ps))
        end
        umeans = ps[:, 1]
        t = minimum(ps[:, 2])
        for i in axes(u, 1)
            _u = view(u, i, :)
            collo_positive_limiter!(_u, umeans[i], t)
        end
    end

    return nothing
end
