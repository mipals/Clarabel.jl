using AMD

struct PanuaPardisoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    ps::Pardiso.PardisoSolver
    Kfake::SparseMatrixCSC{T}

    function PanuaPardisoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        #NB: ignore Dsigns here because pardiso doesn't
        #use information about the expected signs



        #make a pardiso object and perform logical factor
        ps = Pardiso.PardisoSolver()
        Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
        Pardiso.pardisoinit(ps)
        Pardiso.set_phase!(ps, Pardiso.ANALYSIS)
        Pardiso.fix_iparm!(ps, :N)

        # Turn off IR from Clarabel. Instead use the IR one from directly from Panua. 
        # Some caveats: https://github.com/oxfordcontrol/Clarabel.jl/issues/161
        settings.iterative_refinement_enable = false
        Pardiso.set_iparm!(ps,8,settings.iterative_refinement_max_iter)

        # PanuaPardiso can handle rank deficient problems, so no need to regularize
        # settings.static_regularization_enable = false  

        # Set permuation
        # perm = amd(KKT)
        # Pardiso.set_perm!(ps, perm)
        
        # Debugging parameters
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

    # MKL is quite robust and will usually produce some
    # kind of factorization unless there is an explicit
    # zero pivot or some other nastiness.   "success"
    # here just means that it didn't fail outright, although
    # the factorization could still be garbage

    # Recompute the numeric factorization using fake RHS
    try
        Pardiso.set_phase!(ldlsolver.ps, Pardiso.NUM_FACT)
        Pardiso.pardiso(ldlsolver.ps, K, [1.])
        # ldlsolver.Kfake = K
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

    # We don't need the KKT system here since it is already
    #factored, but Pardiso still wants an argument with the
    #correct dimension.   It seems happy for us to pass a
    #placeholder with (almost) no data in it though.
    Kfake = ldlsolver.Kfake

    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ps, x, Kfake, b)

end
