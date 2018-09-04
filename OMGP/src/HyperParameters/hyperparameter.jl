#== HyperParameter Type ==#

mutable struct HyperParameter{T<:Real}
    value::Base.RefValue{T}
    interval::Interval{T}
    fixed::Bool
    opt::Optimizer
    # function HyperParameter{T}(x::T,I::Interval{T};fixed::Bool=false,opt::Optimizer=Momentum(η=0.01)) where {T<:Real}
    function HyperParameter{T}(x::T,I::Interval{T};fixed::Bool=false,opt::Optimizer=Adam(α=0.1)) where {T<:Real}
    # function HyperParameter{T}(x::T,I::Interval{T};fixed::Bool=false,opt::Optimizer=VanillaGradDescent(η=0.001)) where {T<:Real}
        checkvalue(I, x) || error("Value $(x) must be in range " * string(I))
        new{T}(Ref(x), I, fixed, opt)
    end
end
HyperParameter(x::T, I::Interval{T} = interval(T); fixed::Bool = false, opt::Optimizer = Adam(α=0.01)) where {T<:Real} = HyperParameter{T}(x, I, fixed, opt)

eltype(::HyperParameter{T}) where {T} = T

@inline getvalue(θ::HyperParameter{T}) where {T}= getindex(θ.value)

function setvalue!(θ::HyperParameter{T}, x::T) where {T}
    checkvalue(θ.interval, x) || error("Value $(x) must be in range " * string(θ.interval))
    setindex!(θ.value, x)
    return θ
end

checkvalue(θ::HyperParameter{T}, x::T) where {T} = checkvalue(θ.interval, x)

convert(::Type{HyperParameter{T}}, θ::HyperParameter{T}) where {T<:Real} = θ
function convert(::Type{HyperParameter{T}}, θ::HyperParameter) where {T<:Real}
    HyperParameter{T}(convert(T, getvalue(θ)), convert(Interval{T}, θ.bounds))
end

function show(io::IO, θ::HyperParameter{T}) where {T}
    print(io, string("HyperParameter(", getvalue(θ), ",", string(θ.interval), ")"))
end

gettheta(θ::HyperParameter) = theta(θ.interval, getvalue(θ))

settheta!(θ::HyperParameter, x::T) where {T}= setvalue!(θ, eta(θ.interval,x))

checktheta(θ::HyperParameter, x::T) where {T} = checktheta(θ.interval, x)

getderiv_eta(θ::HyperParameter) = deriv_eta(θ.interval, getvalue(θ))

for op in (:isless, :(==), :+, :-, :*, :/)
    @eval begin
        $op(θ1::HyperParameter, θ2::HyperParameter) = $op(getvalue(θ1), getvalue(θ2))
        $op(a::Number, θ::HyperParameter) = $op(a, getvalue(θ))
        $op(θ::HyperParameter, a::Number) = $op(getvalue(θ), a)
    end
end
###Old version not using reparametrization
update!(param::HyperParameter{T},grad::T) where {T} = begin
    # println("Correc : $(getderiv_eta(param)), Grad : $(GradDescent.update(param.opt,grad)), theta : $(gettheta(param))")
    isfree(param) ? settheta!(param, gettheta(param) + update(param.opt,getderiv_eta(param)*grad)) : nothing
    # isfree(param) ? setvalue!(param, getvalue(param) + GradDescent.update(param.opt,grad)) : nothing
end

isfree(θ::HyperParameter) = !θ.fixed

setfixed!(θ::HyperParameter) = θ.fixed = true



setfree!(θ::HyperParameter) = θ.fixed = false

mutable struct HyperParameters{T<:AbstractFloat}
    hyperparameters::Array{HyperParameter{T},1}
    function HyperParameters{T}(θ::Vector{T},intervals::Array{Interval{T,A,B}}) where {A<:Bound{T},B<:Bound{T}} where {T<:AbstractFloat}
        this = new(Vector{HyperParameter{T}}())
        for (val,int) in zip(θ,intervals)
            push!(this.hyperparameters,HyperParameter{T}(val,int))
        end
        return this
    end
end
function HyperParameters(θ::Vector{T},intervals::Vector{Interval{T,A,B}}) where {A<:Bound{T},B<:Bound{T}} where {T<:Real}
    HyperParameters{T}(θ,intervals)
end

@inline getvalue(θ::HyperParameters) = broadcast(getvalue,θ.hyperparameters)

function Base.getindex(p::HyperParameters,it::Integer)
    return p.hyperparameters[it]
end

function update!(param::HyperParameters,grad)
    for i in 1:length(param.hyperparameters)
        update!(param.hyperparameters[i],grad[i])
    end
end

setfixed!(θ::HyperParameters) = setfixed!.(θ.hyperparameters)

setfree!(θ::HyperParameters) = setfree!.(θ.hyperparameters)
