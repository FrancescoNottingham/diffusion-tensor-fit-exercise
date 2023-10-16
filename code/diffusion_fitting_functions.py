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
        
    G_values = np.array([bvecs[0,:]**2,bvecs[1,:]**2,bvecs[2,:]**2, 2*bvecs[1,:]*bvecs[0,:], 2*bvecs[2,:]*bvecs[1,:], 2*bvecs[0,:]*bvecs[2,:]])

    return np.transpose(G_values)

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


def get_s0_value(s0,x,y,z):
    return s0[x,y,z]

def remove_b0s(data,bvals,bvecs):

    index = np.nonzero(bvals)[0]
    new_data = data[...,index]
    new_bvals = bvals[index]
    new_bvecs = bvecs[:,index]
    
    return new_data, new_bvals, new_bvecs

def extract_non0_data(data, bvals, bvecs, x,y,z):

    timeseries_all = data[x,y,z,:]
    index = np.nonzero(timeseries_all)[0]
    timeseries = data[x,y,z,index]
    new_bvals = bvals[index]
    new_bvecs = bvecs[:,index]

    return timeseries, new_bvals, new_bvecs

def calculate_C_vals(timeseries,s0_value):
    ''' Calculates C values from Stam's thesis paper (-ln(S_k/s0))
    '''
    return -np.log(timeseries/s0_value)
    
def calculate_diffusion_values(timeseries,s0_value,bvals,bvecs):
    ''' Returns the diffusion tensor values (X) in the following order:
    [Dxx,Dyy,Dzz,Dxy,Dyz,Dxz]

    The calculation is as follows:
    (At.A)^-1 . At.C
    where At is the transpose of A
    '''

    A = create_matrix_A(bvals,bvecs)
    C = calculate_C_vals(timeseries,s0_value)

    AT_dot_A_inv = np.linalg.inv(np.dot(np.transpose(A),A))
    AT_dot_C = np.transpose(np.dot(C,A))
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


    return np.linalg.eigvalsh(diff_tensor)

def calculate_fractional_anisotropy(evals):
    ''' Calculates fractional anisotropy given the eigenvalues of a diffusion tensor
    ''' 
    l_1 = evals[0]
    l_2 = evals[1]
    l_3 = evals[2]

    return np.sqrt(1-(l_1*l_2 + l_2*l_3 + l_3*l_1)/(l_1**2 + l_2**2 + l_3**2))

def diffusion_tensor_fit(diffusion_data,brain_mask,bvals,bvecs):

    fractional_anisotropy = np.zeros(diffusion_data.shape[:-1])

    s0 = create_matrix_s0(diffusion_data,bvals)

    diffusion_data_no_b0, bvals_no_b0, bvecs_no_b0 = remove_b0s(diffusion_data,bvals,bvecs)

    for x in np.arange(diffusion_data_no_b0.shape[0]):
        for y in np.arange(diffusion_data_no_b0.shape[1]):
            for z in np.arange(diffusion_data_no_b0.shape[2]):

                if brain_mask[x,y,z] == 0:
                    fractional_anisotropy[x,y,z] = 0
                else:
                    s0_value = get_s0_value(s0,x,y,z)
                    timeseries_no0, bvals_no0, bvecs_no0 = extract_non0_data(diffusion_data_no_b0,bvals_no_b0,bvecs_no_b0, x,y,z)
                    if timeseries_no0.size < 6:
                        fractional_anisotropy[x,y,z] = 0
                    else:
                        diffusion_tensor_values = calculate_diffusion_values(timeseries_no0,s0_value,bvals_no0,bvecs_no0)
                        evals = calculate_diffusion_eigenvalues(diffusion_tensor_values)
                        fractional_anisotropy[x,y,z] = calculate_fractional_anisotropy(evals)

    return fractional_anisotropy
