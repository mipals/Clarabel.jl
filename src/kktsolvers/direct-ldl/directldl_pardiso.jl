import Pardiso

abstract type AbstractPardisoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}  end

# MKL Pardiso variant
struct MKLPardisoDirectLDLSolver{T} <: AbstractPardisoDirectLDLSolver{T}
    ps::Pardiso.MKLPardisoSolver

    function MKLPardisoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}
        Pardiso.mkl_is_available() || error("MKL Pardiso is not available")
        ps = Pardiso.MKLPardisoSolver()
        pardiso_init(ps,KKT,Dsigns,settings)
        return new(ps)
    end
end

# Panua Pardiso variant
struct PanuaPardisoDirectLDLSolver{T} <: AbstractPardisoDirectLDLSolver{T}
    ps::Pardiso.PardisoSolver
    # In order to avoid re-casting the indexing to i32 
    colptr::Vector{Int32} 
    rowval::Vector{Int32}

    function PanuaPardisoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        Pardiso.panua_is_available() || error("Panua Pardiso is not available")
        ps = Pardiso.PardisoSolver()
        pardiso_init(ps,KKT,Dsigns,settings)
        ps.iparm[8]=-99 # No IR
        # Save colptr and rowval in Int32 as is required by Panua. This make is avoid allocations for each call
        colptr = convert(Vector{Int32}, KKT.colptr) 
        rowval = convert(Vector{Int32}, KKT.rowval)
        return new(ps, colptr, rowval)
    end
end

function pardiso_init(ps,KKT,Dsigns,settings)

        #NB: ignore Dsigns here because pardiso doesn't
        #use information about the expected signs

        #perform logical factor
        Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
        Pardiso.pardisoinit(ps)
        Pardiso.fix_iparm!(ps, :N)
        Pardiso.set_phase!(ps, Pardiso.ANALYSIS)
        Pardiso.pardiso(ps, KKT, [1.])  #RHS irrelevant for ANALYSIS
end 



DirectLDLSolversDict[:mkl]   = MKLPardisoDirectLDLSolver
DirectLDLSolversDict[:panua] = PanuaPardisoDirectLDLSolver
required_matrix_shape(::Type{PanuaPardisoDirectLDLSolver}) = :tril
required_matrix_shape(::Type{MKLPardisoDirectLDLSolver}) = :tril

#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::AbstractPardisoDirectLDLSolver{T},
    index::AbstractVector{DefaultInt},
    values::Vector{T}
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!
end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    ldlsolver::AbstractPardisoDirectLDLSolver{T},
    index::AbstractVector{DefaultInt},
    scale::T
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!
end


#refactor the linear system
function refactor!(ldlsolver::AbstractPardisoDirectLDLSolver{T},K::SparseMatrixCSC{T}) where{T}

    # Pardiso is quite robust and will usually produce some 
    # kind of factorization unless there is an explicit 
    # zero pivot or some other nastiness.   "success" 
    # here just means that it didn't fail outright, although 
    # the factorization could still be garbage 

    # Recompute the numeric factorization susing fake RHS
    try 
        ps = ldlsolver.ps
        Pardiso.set_phase!(ps, Pardiso.NUM_FACT)
        # Pardiso.pardiso(ps, K, [1.])
        ERR = Ref{Int32}(0)
        ccall(Pardiso.pardiso_f[], Cvoid,
            (Ptr{Int}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32},
            Ptr{Int32}, Ptr{T}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32},
            Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{T}, Ptr{T},
            Ptr{Int32}, Ptr{Float64}),
            ps.pt, Ref(ps.maxfct), Ref(Int32(ps.mnum)), Ref(Int32(ps.mtype)), Ref(Int32(ps.phase)),
            Ref(Int32(size(K,1))), K.nzval, ldlsolver.colptr, ldlsolver.rowval, ps.perm,
            Ref(Int32(1)), ps.iparm, Ref(Int32(ps.msglvl)), [one(T)], [one(T)],
            ERR, ps.dparm)
        Pardiso.check_error(ps, ERR[])
        return is_success = true
    catch 
        return is_success = false
    end
     
end


#solve the linear system
function solve!(
    ldlsolver::AbstractPardisoDirectLDLSolver{T},
    KKT::SparseMatrixCSC{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    ps  = ldlsolver.ps
    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    # Pardiso.pardiso(ps, x, KKT, b)
    ERR = Ref{Int32}(0)
    ccall(Pardiso.pardiso_f[], Cvoid,
        (Ptr{Int}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32},
        Ptr{Int32}, Ptr{T}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32},
        Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{T}, Ptr{T},
        Ptr{Int32}, Ptr{Float64}),
        ps.pt, Ref(ps.maxfct), Ref(Int32(ps.mnum)), Ref(Int32(ps.mtype)), Ref(Int32(ps.phase)),
        Ref(Int32(length(x))), KKT.nzval, ldlsolver.colptr, ldlsolver.rowval, ps.perm,
        Ref(Int32(1)), ps.iparm, Ref(Int32(ps.msglvl)), b, x,
        ERR, ps.dparm)
    Pardiso.check_error(ps, ERR[])
end
