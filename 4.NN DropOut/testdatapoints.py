import numpy as np

def set_seed(seed):
    np.random.seed(seed)



def labeled_two_classPoints(Aamount=25, Bamount=25,                            
                            Aweight=1,   Bweight=1,
                            scale=5):
    zero_centr = np.array([-0.5,-0.5])
    A_centr = (np.random.rand(2) + zero_centr)*2*scale
    B_centr = (np.random.rand(2) + zero_centr)*2*scale
    
    get_unit_vector = lambda rad: np.c_[np.cos(rad), np.sin(rad)]

    A_magnitude    = np.random.randn(Aamount)*Aweight
    A_angles       = np.random.rand(Aamount)*2*np.pi
    A_unit_vectors = get_unit_vector(A_angles)
    A_deviations   = A_unit_vectors*A_magnitude[:,np.newaxis]
    A = A_deviations + A_centr

    B_magnitude    = np.random.randn(Bamount)*Bweight
    B_angles       = np.random.rand(Bamount)*2*np.pi
    B_unit_vectors = get_unit_vector(B_angles)
    B_deviations   = B_unit_vectors*B_magnitude[:,np.newaxis]
    B = B_deviations + B_centr

    A_labeled = np.c_[A, np.ones(A.shape[0])]
    B_labeled = np.c_[B, np.zeros(B.shape[0])]


    result = np.row_stack((A_labeled,B_labeled))

    return result


if __name__ == '__main__':
    print(labeled_two_classPoints())


