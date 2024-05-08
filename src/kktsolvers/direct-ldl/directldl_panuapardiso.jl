struct PanuaPardisoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    ps::Pardiso.PardisoSolver
    KKT::SparseMatrixCSC{T}

    function PanuaPardisoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        #NB: ignore Dsigns here because PanuaPardiso doesn't
        #use information about the expected signs

        #make a PanuaPardiso object and perform symbolic factorization
        ps = Pardiso.PardisoSolver()
        Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
        Pardiso.pardisoinit(ps)
        Pardiso.set_phase!(ps, Pardiso.ANALYSIS)
        Pardiso.fix_iparm!(ps, :N)

        # Completely turn of iterative refinement from Panua. 
        Pardiso.set_iparm!(ps,8,-99)

        # Possible to specify own permuation. For now let Panua use its internal ordering (METIS)
        # perm = amd(KKT)
        # Pardiso.set_perm!(ps, perm)
        
        # For debugging purposes you can turn on the messaging
        # Pardiso.set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)

        # Create symbolic factorization with fake rhs
        Pardiso.pardiso(ps, KKT, [1.])  #RHS irrelevant for ANALYSIS

        return new(ps,KKT)
    end
end

DirectLDLSolversDict[:panua] = PanuaPardisoDirectLDLSolver
required_matrix_shape(::Type{PanuaPardisoDirectLDLSolver}) = :tril

#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::PanuaPardisoDirectLDLSolver{T},
    index::AbstractVector{Int},
    values::Vector{T}
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!

end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    ldlsolver::PanuaPardisoDirectLDLSolver{T},
    index::AbstractVector{Int},
    scale::T
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!
end


#refactor the linear system
function refactor!(ldlsolver::PanuaPardisoDirectLDLSolver{T},K::SparseMatrixCSC{T}) where{T}

    # Recompute the numeric factorization using fake RHS
    try
        Pardiso.set_phase!(ldlsolver.ps, Pardiso.NUM_FACT)
        Pardiso.pardiso(ldlsolver.ps, K, [1.])
        return is_success = true
    catch
        return is_success = false
    end

end


#solve the linear system
function solve!(
    ldlsolver::PanuaPardisoDirectLDLSolver{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    ps  = ldlsolver.ps

    # We need to pass KKT to pardiso even if it already factored.
    #the reason being that the matrix is used for iterative refinement.
    #note that we for now disable IR and let Clarabel handle that, so in this 
    #case the matrix does not matter (as no IR is performed in PanuaPardiso)
    KKT = ldlsolver.KKT

    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ps, x, KKT, b)

end
