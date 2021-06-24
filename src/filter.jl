"""
Filter function for polynomial chaos

- @args p: λ, Δ, ℓ, op, ..., mode

"""
function Base.filter!(u::T, p...) where {T<:AbstractArray{<:AbstractFloat,1}}

    q0 = eachindex(u) |> first
    q1 = eachindex(u) |> last

    λ = p[1]
    mode = p[end]

    if mode == :l2
        if length(p) > 2
            Δ = p[2]
            op = p[3]
            filter_l2!(u, op, λ, Δ)
        else
            filter_l2!(u, λ)
        end
    elseif mode == :l1
        for i = q0+1:q1
            sc = 1.0 - 5.0 * λ * i * (i - 1) * ℓ[i] / abs(u[i])
            if sc < 0.0
                sc = 0.0
            end
            u[i] *= sc
        end
    elseif mode == :lasso
        nr = length(u)
        _λ = abs(u[end]) / (nr * (nr - 1) * ℓ[end])
        for i = q0+1:q1
            sc = 1.0 - _λ * i * (i - 1) * ℓ[i] / abs(u[i] + 1e-8)
            if sc < 0.0
                sc = 0.0
            end
            u[i] *= sc
        end
    elseif mode == :dev
        nr = length(u)
        for i = q0+1:q1
            _λ = λ * abs(u[i] / (u[1] + 1e-10)) / (nr * (nr - 1) * ℓ[i])
            u[i] /= (1.0 + _λ * i^2 * (i - 1)^2)
        end
    else
        throw("unavailable filter mode")
    end

end

function Base.filter!(u::AbstractArray{<:AbstractFloat,2}, dim::Integer, p...)

    if dim == 1
        for j in axes(u, 2)
            _u = @view u[:, j]
            filter!(_u, p...)
        end
    elseif dim == 2
        for i in axes(u, 1)
            _u = @view u[i, :]
            filter!(_u, p...)
        end
    end

end

function Base.filter!(u::AbstractArray{<:AbstractFloat,3}, dim::Integer, p...)

    if dim == 1
        for k in axes(u, 3), j in axes(u, 2)
            _u = @view u[:, j, k]
            filter!(_u, p...)
        end
    elseif dim == 2
        for k in axes(u, 3), i in axes(u, 1)
            _u = @view u[i, :, k]
            filter!(_u, p...)
        end
    elseif dim == 3
        for j in axes(u, 2), i in axes(u, 1)
            _u = @view u[i, j, :]
            filter!(_u, p...)
        end
    end

end

function filter_l2!(u::T, λ) where {T<:AbstractArray{<:AbstractFloat,1}}

    q0 = eachindex(u) |> first
    q1 = eachindex(u) |> last

    for i = q0+1:q1
        u[i] /= (1.0 + λ * i^2 * (i - 1)^2)
    end

    return nothing

end

function filter_l2!(
    u::T1,
    op::T2,
    λ,
    Δ,
) where {T1<:AbstractArray{<:AbstractFloat,1},T2<:AbstractOrthoPoly}

    q0 = eachindex(u) |> first
    q1 = eachindex(u) |> last

    uRan = evaluatePCE(u, op.quad.nodes, op)
    δ = 0.5 * (maximum(uRan) - minimum(uRan))
    _λ = λ * (exp(δ / Δ) - 1.0)
    #_λ = λ * δ / Δ

    for i = q0+1:q1
        u[i] /= (1.0 + _λ * i^2 * (i - 1)^2)
    end

    return nothing

end
