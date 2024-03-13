using Distributed
addprocs(31, exeflags="--project=$(Base.active_project())")

using ACEpotentials
using ExtXYZ
using JSON3
using StaticArrays
using PyCall
using ASE
using JuLIP: Atoms, JData
using LinearAlgebra

const ase_io    = PyNULL()
copy!(ase_io, pyimport_conda("ase.io", "ase", "rmg"))

function ase2JuLIPat(ase_atom, inv_poly_idx)
    at = Atoms(ase_atom)
    at.data = Dict("energy" => JData{Float64}(Inf, 0.0 ,ase_atom.po.get_potential_energy()[inv_poly_idx]))
    at.cell = SMatrix{3,3}(I .* 20.0)
    return at
end


# === IOs ===
data_file = "ch2nh2_esps_cheb_esps_fix.xyz"

# read as ase atoms
ASE_atoms = ASEAtoms.(ase_io.read("ch2nh2_esps_cheb_esps.xyz", index=":"))

# wrap as JuLIP Atoms, each with energy = f_{1, k}
JuLIP_atoms_f31 = ase2JuLIPat.(ASE_atoms, 1)
JuLIP_atoms_f32 = ase2JuLIPat.(ASE_atoms, 2)
JuLIP_atoms_f33 = ase2JuLIPat.(ASE_atoms, 3)

##

# now we do ACE, we use three models for three invariant polynomials
# TODO: thank about many body expansion of each E_k? 
# note
# deg(E1 + E2 + E3) ≤ deg(E1E2 + E1E3 + E2E3) ≤ deg(E1E2E3)
# probably need a small, medium and larger model
# we probably still need the 2 body paired interaction part?
# 

# getting okay errors for f31 and f32 but f33 is problematic
model_f31 = acemodel(elements = [:C, :N, :H],
					  order = 2,
					  totaldegree = 6,
					  rcut = 10.0)

model_f32 = acemodel(elements = [:C, :N, :H],
                    order = 2,
                    totaldegree = 8,
                    rcut = 10.0)


model_f33 = acemodel(elements = [:C, :N, :H],
                    order = 3,
                    totaldegree = 10,
                    rcut = 10.0)


## 

using Random

# solver = ACEfit.LSQR(damp = 1e-2, atol = 1e-6);
solver = ACEfit.BLR(factorization=:svd) 

# how does it changes the meaning of prior??
P = smoothness_prior(model_f31; p = 4)

# === 1. fitting f_31 ===

Random.seed!(123)

train_ratio = 0.8
Ntrain = floor(Int, length(JuLIP_atoms_f31) * train_ratio)
shuffle!(JuLIP_atoms_f31)

train_f31, valid_f31 = JuLIP_atoms_f31[1:Ntrain], JuLIP_atoms_f31[Ntrain+1:end]

P = smoothness_prior(model_f32; p = 4)
acefit!(model_f32, train_f32; solver=solver, prior = P);

@info("Training error")
ACEpotentials.linear_errors(train_f32, model_f32);

@info("Testing error")
err = ACEpotentials.linear_errors(valid_f32, model_f32);


# === 2. fitting f_32 ===
Random.seed!(123)

train_ratio = 0.8
Ntrain = floor(Int, length(JuLIP_atoms_f32) * train_ratio)
shuffle!(JuLIP_atoms_f32)

train_f32, valid_f32 = JuLIP_atoms_f32[1:Ntrain], JuLIP_atoms_f32[Ntrain+1:end]

acefit!(model_f32, train_f32; solver=solver, prior = P);

@info("Training error")
ACEpotentials.linear_errors(train_f32, model_f32);

@info("Testing error")
err = ACEpotentials.linear_errors(valid_f32, model_f32);

# === 3. fitting f_33 ===
Random.seed!(123)

train_ratio = 0.8
Ntrain = floor(Int, length(JuLIP_atoms_f33) * train_ratio)
shuffle!(JuLIP_atoms_f33)

train_f33, valid_f33 = JuLIP_atoms_f33[1:Ntrain], JuLIP_atoms_f33[Ntrain+1:end]

acefit!(model_f33, train_f33; solver=solver, prior = P);

@info("Training error")
ACEpotentials.linear_errors(train_f33, model_f33);

@info("Testing error")
err = ACEpotentials.linear_errors(valid_f33, model_f33);