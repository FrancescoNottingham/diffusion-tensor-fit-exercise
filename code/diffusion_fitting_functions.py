#functions for task

#importing section
import numpy as np

def create_bval_array(bvals):
    ''' Creates a matrix containing the b values in the correct order for manipulation in
    the A matrix in equation 2.11 of Stam's thesis

    For each element in the bvals vector a row of 6 values is filled in, for example:
    create_bval_array([0,1,2])
    returns
    [[0,0,0,0,0,0],
    [1,1,1,1,1,1],
    [2,2,2,2,2,2]]
    '''
    return np.transpose(np.tile(bvals, (6,1)))

def create_bvec_array(bvecs):
    ''' Creates a matrix containing the gradients in the order outlined in equation 2.11 of 
    Stam's thesis

    For each 3D bvec (g1,g2,g3) we get the following row:
    [g1^2,g2^2,g3^2,2*g1*g2,2*g2*g3,2*g3*g1]

    The function iterates over all bvecs to give a matrix which is 6 colums and N rows (number
    of bvecs - the column number)
    '''

    G_values = np.zeros((bvecs.shape[1],6))
    
    for i in np.arange(bvecs.shape[1]):
        G1 = np.tile(bvecs[:,i], (1,2))
        G2 = np.append(bvecs[:,i], [2*bvecs[1,i], 2*bvecs[2,i], 2*bvecs[0,i]])
        G_values[i,:] = G1*G2
    return G_values

def create_matrix_A(bvals,bvecs):
    ''' Creates the A matrix in equation 2.11 of Stam's thesis
    '''

    G_matrix = create_bvec_array(bvecs)
    b_matrix = create_bval_array(bvals)
    return G_matrix*b_matrix

def create_matrix_s0(data,bvals):
    ''' Returns a volume with the mean intensities of all diffusion volumes with b=0)
    '''
    return np.mean(data[...,np.where(bvals==0)[0]],axis=data.ndim-1)

def calculate_C_vals(data,bvals,x,y,z):
    ''' Calculates C values from Stam's thesis paper (-ln(S_k/s0))
    '''

    s0 = create_matrix_s0(data,bvals)

    C_vals = np.zeros(data.shape[-1])
    if s0[x,y,z] == 0:
        return C_vals
    else:
        for i in np.arange(0,data.shape[-1]):
            C_vals[i] = -np.log(data[x,y,z,i]/s0[x,y,z])
        return C_vals
    
def calculate_diffusion_values(data,bvals,bvecs,x,y,z):
    ''' Returns the diffusion tensor values (X) in the following order:
    [Dxx,Dyy,Dzz,Dxy,Dyz,Dxz]

    The calculation is as follows:
    (At.A)^-1 . At.C
    where At is the transpose of A
    '''

    A = create_matrix_A(bvals,bvecs)
    C = calculate_C_vals(data,bvals,x,y,z)

    AT_dot_A_inv = np.linalg.inv(np.dot(np.transpose(A),A))
    AT_dot_C = np.dot(np.transpose(A),C)
    return np.dot(AT_dot_A_inv,AT_dot_C)

def calculate_diffusion_eigenvalues(X):
    ''' Returns the diffusion tensor eigenvalues and eigenvectors given 
    values (X) in the following order:
    [Dxx,Dyy,Dzz,Dxy,Dyz,Dxz]
    '''

    diff_tensor = np.zeros((3,3))
    
    diff_tensor[0,0] = X[0]
    diff_tensor[1,1] = X[1]
    diff_tensor[2,2] = X[2]
    diff_tensor[0,1] = X[3] 
    diff_tensor[1,0] = X[3]
    diff_tensor[1,2] = X[4] 
    diff_tensor[2,1] = X[4]
    diff_tensor[0,2] = X[5] 
    diff_tensor[2,0] = X[5]

    eval, evec = np.linalg.eig(diff_tensor)

    return eval, evec

def calculate_fractional_anisotropy(evals):
    ''' Calculates fractional anisotropy given the eigenvalues of a diffusion tensor
    ''' 
    l_1 = evals[0]
    l_2 = evals[1]
    l_3 = evals[2]

    return np.sqrt(1-(l_1*l_2 + l_2*l_3 + l_3*l_1)/(l_1**2 + l_2**2 + l_3**2))

def diffusion_tensor_fit(diffusion_data,brain_mask,bvals,bvecs):

    fractional_anisotropy = np.zeros(diffusion_data.shape)

    for x in np.arange(diffusion_data.shape[0]):
        for y in np.arange(diffusion_data.shape[1]):
            for z in np.arange(diffusion_data.shape[2]):
                if brain_mask[x,y,z] == 0 or np.all(diffusion_data[x,y,z,:])==0:
                    fractional_anisotropy[x,y,z] = 0
                else:
                    diffusion_tensor_values = calculate_diffusion_values(diffusion_data,bvals,bvecs,x,y,z)
                    evals, evecs = calculate_diffusion_eigenvalues(diffusion_tensor_values)
                    fractional_anisotropy[x,y,z] = calculate_fractional_anisotropy(evals)

    return fractional_anisotropy


    