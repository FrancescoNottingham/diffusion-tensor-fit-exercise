#tests for task functions

from diffusion_fitting_functions import *
import numpy as np

def test_create_bval_array():
    test_array1 = np.array([0,1,2])
    test_solution1 = np.array([[0,0,0,0,0,0],
                              [1,1,1,1,1,1],
                              [2,2,2,2,2,2]])
    
    assert (create_bval_array(test_array1) == test_solution1).all

def test_create_bvec_array():
    test_array2 = np.array([[0,1,3],[0,2,2],[0,3,1]])
    test_solution2 = np.array([[0,0,0,0,0,0],
                               [1,4,9,4,12,6],
                               [9,4,1,12,4,6]])

    assert (create_bvec_array(test_array2) == test_solution2).all

def test_create_matrix_A():

    test_array1 = np.array([0,1,2])
    test_array2 = np.array([[0,1,3],[0,2,2],[0,3,1]])

    test_solution3 = np.array([[0,0,0,0,0,0],
                               [1,4,9,4,12,6],
                               [18,8,2,24,8,12]])
    
    assert (test_solution3 == create_matrix_A(test_array1,test_array2)).all
    assert (test_solution3.shape == create_matrix_A(test_array1,test_array2).shape)

def test_create_matrix_s0():
    bval_test = np.array([0,1,0,2,3,0,0])
    data_test = np.array([4,0,4,0,0,4,4])

    assert create_matrix_s0(data_test,bval_test) == 4

def test_remove_b0s():
    
    data = np.array([[5,1,3],[4,2,2],[2,3,1]])
    bvals = np.array([1,0,3])
    bvecs = np.array([[0,1,3],[0,2,2],[0,3,1]])

    data, bvals, bvecs = remove_b0s(data,bvals,bvecs)

    assert (data == np.array([[5,3],[4,2],[2,1]])).all
    assert (bvals == np.array([1,3])).all
    assert (bvecs == np.array([[0,3],[0,2],[0,1]])).all

def test_extract_non0_data():
    x = 0
    y = 0
    z = 0
    data = np.array([[[[1,1],[1,1]],[[1,1],[1,1]]],
                     [[[0,0],[0,0]],[[0,0],[0,0]]]])
    bvals = np.array([2,7])
    bvecs = np.array([[5,3],[3,2],[7,1]])

    timeseries, bvals, bvecs = extract_non0_data(data, bvals, bvecs, x,y,z)

    assert (timeseries == np.array([1])).all
    assert (bvals == np.array([2])).all
    assert (bvecs == np.array([[5],[3],[7]])).all