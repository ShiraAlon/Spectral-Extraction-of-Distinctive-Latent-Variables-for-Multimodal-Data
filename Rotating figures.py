

exec(open("Functions.py").read())
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.linalg import eig, svd
csfont = {'fontname':'Times New Roman'}
import matplotlib.font_manager as font_manager
font = font_manager.FontProperties(family='Times New Roman',size=24)
from deeptime.decomposition import KernelCCA
from deeptime.kernels import GaussianKernel


def Differential_method(X1, X2, gt_Y, gt_R, K=200, k=30):
    """ Function to compute Differential Method.
    X1, X2 are numpy arrays containing the two datsets.
    gt_Y, gt_R are the ground truth vectors for the rotation angle of the Yoda and bunny respectively.
    K is the number of neighbors to consider in the kernel's adaptive bandwidth.
    k is the number of neighbors to vectors we use in the filtering"""

    # Differential vectors method
    K1 = Kernel_matrix(X1, K)
    K2 = Kernel_matrix(X2, K)

    L1, d1, v1 = LG_sym(K1)
    L2, d2, v2 = LG_sym(K2)

    s, u1 = calc_differential_vec(L2, v1[:, 1:], k)
    s, u2 = calc_differential_vec(L1, v2[:, 1:], k)

    # Computing the correlation of the leading differential vector with the ground truth
    corr_DV_R = np.max(np.abs(circ_convolution(u1[:, 1], gt_R)))
    corr_DV_Y = np.max(np.abs(circ_convolution(u2[:, 1], gt_Y)))

    # Computing the signal to noise ratio of the leading differential vector
    sig_noise_DV_R = calc_sig_to_noise(u1[:, 1], gt_R, 20, False)
    sig_noise_DV_Y = calc_sig_to_noise(u2[:, 1], gt_Y, 20, False)
    return corr_DV_Y, corr_DV_R, sig_noise_DV_Y, sig_noise_DV_R

def CCA_method(X1, X2, gt_Y, gt_R):
    """ Function to compute KCCA Method.
    X1, X2 are numpy arrays containing the two datsets.
    gt_Y, gt_R are the ground truth vectors for the rotation angle of the Yoda and bunny respectively"""

    sigma = 1
    kernel = GaussianKernel(sigma)

    kcca_estimator = KernelCCA(kernel, n_eigs=9, epsilon=1e-3)
    kcca_model = kcca_estimator.fit((X1, X2)).fetch_model()
    ev_real = np.real(kcca_model.eigenvectors)

    proj_data = np.eye(X1.shape[0]) - ev_real @ ev_real.T
    X1_c = proj_data @ X1
    X2_c = proj_data @ X2

    v_CCA_A, d_CCA_A, _ = np.linalg.svd(X1_c)
    v_CCA_B, d_CCA_B, _ = np.linalg.svd(X2_c)

    # Computing the correlation of the CCA vector with the ground truth
    corr_CCA_R = np.max(np.abs(circ_convolution(v_CCA_B[:, 1], gt_R)))
    corr_CCA_Y = np.max(np.abs(circ_convolution(v_CCA_A[:, 1], gt_Y)))

    # Computing the signal to noise ratio of the CCA vector
    sig_noise_CCA_R = calc_sig_to_noise(v_CCA_B[:, 1], gt_R, 20, False)
    sig_noise_CCA_Y = calc_sig_to_noise(v_CCA_A[:, 1], gt_Y, 20, False)

    return corr_CCA_Y, corr_CCA_R, sig_noise_CCA_Y, sig_noise_CCA_R

def FKT_method(X1, X2, gt_Y, gt_R, K=200):
    """ Function to compute FKT Method.
    X1, X2 are numpy arrays containing the two datsets.
    gt_Y, gt_R are the ground truth vectors for the rotation angle of the Yoda and bunny respectively.
    K is the number of neighbors to consider in the kernel's adaptive bandwidth"""

    # Differential vectors method
    K1 = Kernel_matrix(X1, K)
    K2 = Kernel_matrix(X2, K)

    # Computing a regular laplacian matrix
    G1 = np.diag(np.sum(K1, axis=0)) - K1
    G2 = np.diag(np.sum(K2, axis=0)) - K2

    # Adding some noise
    M1 = G1 + 10**(-7) * np.eye(G1.shape[0])
    M2 = G2 + 10**(-7) * np.eye(G2.shape[0])

    # The FKT operator
    FK1 = np.linalg.inv(M1 + M2) @ M1
    FK2 = np.linalg.inv(M1 + M2) @ M2

    # Computing the eigen decomposition of the FKT operators
    FK_values_1, eig_vec_FK_1 = eig(FK1)
    FK_values_2, eig_vec_FK_2 = eig(FK2)

    # Sorting the eigenvectors
    idx_1 = np.argsort(FK_values_1)[::-1]
    eig_vec_FK_1 = eig_vec_FK_1[:, idx_1]

    idx_2 = np.argsort(FK_values_2)[::-1]
    eig_vec_FK_2 = eig_vec_FK_2[:, idx_2]

    # Computing the correlation of the FKT vector with the ground truth
    corr_FK_R = np.max(np.abs(circ_convolution(eig_vec_FK_1[:, 0], gt_R)))
    corr_FK_Y = np.max(np.abs(circ_convolution(eig_vec_FK_2[:, 0], gt_Y)))

    # Computing the signal to noise ratio of the FKT vector
    sig_noise_FK_R = calc_sig_to_noise(eig_vec_FK_1[:, 0], gt_R, 20, False)
    sig_noise_FK_Y = calc_sig_to_noise(eig_vec_FK_2[:, 0], gt_Y, 20, False)

    return corr_FK_Y, corr_FK_R, sig_noise_FK_Y, sig_noise_FK_R


def simulation(X1, X2, gt_Y, gt_R,  N = 20,  sig = 0.05):
    """ Adding noise to the datasets, and checking the results of the tree models.
    X1, X2 are numpy arrays containing the datasets of the two modalities.
    gt_Y, gt_R are the ground truth vectors for the rotation angle of the Yoda and bunny respectively.
    N is the number of iterations - in which we add noise to the data and run the tree models.
    sig determines the std of the gaussian noise added to the data.
    """

    DV_results_matrix = np.zeros((N, 4))
    CCA_results_matrix = np.zeros((N, 4))
    FKT_results_matrix = np.zeros((N, 4))

    for i in range(N):
        sig_i = sig * i
        n, p = X1.shape
        # Adding small Gaussian noise to the data
        X_A = X1 + np.random.normal(0, sig_i, [n, p])
        X_B = X2 + np.random.normal(0, sig_i, [n, p])
        DV_results_matrix[i, :] = Differential_method(X_A, X_B, gt_Y, gt_R, K=250, k=2)
        CCA_results_matrix[i, :] = CCA_method(X_A, X_B, gt_Y, gt_R)
        FKT_results_matrix[i, :] = FKT_method(X_A, X_B, gt_Y, gt_R, K=250)

    return DV_results_matrix, CCA_results_matrix, FKT_results_matrix


def plot_results(DV_results_matrix, CCA_results_matrix, FKT_results_matrix, N):
    # Set up a figure twice as tall as it is wide
    fig = plt.figure(figsize=(36, 7))
    plt.subplots_adjust(right=1.05)

    # First subplot
    ax = fig.add_subplot(1, 4, 1)

    ax.imshow(X_A[30, :].reshape((80, 60)).T)
    ax.set_title("(A)", fontsize=42, **csfont, pad=20)
    ax.tick_params(left=False, right=False, labelleft=False,
                   labelbottom=False, bottom=False)

    # Second subplot
    ax = fig.add_subplot(1, 4, 2)
    ax.imshow(X_B[31, :].reshape((80, 60)).T)
    ax.set_title("(B)", fontsize=42, **csfont, pad=20)
    ax.tick_params(left=False, right=False, labelleft=False,
                   labelbottom=False, bottom=False)

    # Third subplot
    ax = fig.add_subplot(1, 4, 3)

    ax.plot(np.linspace(0, N * 0.05, N), np.round(100 * DV_results_matrix[:, 0], 8), label='DV - $\psi_A$',
            linewidth=5)
    ax.plot(np.linspace(0, N * 0.05, N), np.round(100 * DV_results_matrix[:, 1], 8), label='DV - $\psi_B$',
            linewidth=5, linestyle='--')

    ax.plot(np.linspace(0, N * 0.05, N), np.round(100 * CCA_results_matrix[:, 0], 8), label='CCA - $\psi_A$',
            linewidth=5)
    ax.plot(np.linspace(0, N * 0.05, N), np.round(100 * CCA_results_matrix[:, 1], 8), label='CCA - $\psi_B$',
            linewidth=5, linestyle='--')

    ax.plot(np.linspace(0, N * 0.05, N), np.round(100 * FKT_results_matrix[:, 0], 8), label='FKT - $\psi_A$',
            linewidth=5)
    ax.plot(np.linspace(0, N * 0.05, N), np.round(100 * FKT_results_matrix[:, 1], 8), label='FKT - $\psi_B$',
            linewidth=5, linestyle='--')

    ax.set_ylabel("Correlation [%]", fontsize=40, **csfont)
    ax.set_xlabel("$\sigma$", fontsize=40, **csfont)
    ax.set_title("(C)", fontsize=42, **csfont, pad=20)
    plt.xticks(np.round(np.linspace(0, N * 0.05, 5), 2), fontsize=38, rotation=0, **csfont)
    plt.yticks(fontsize=38, rotation=0, **csfont)
    plt.legend(loc="upper right", fontsize=20, prop=font)

    # fourth subplot
    ax = fig.add_subplot(1, 4, 4)

    ax.plot(np.linspace(0, N * 0.05, N), np.log(DV_results_matrix[:, 2]), label='DV - $\psi_A$', linewidth=5)
    ax.plot(np.linspace(0, N * 0.05, N), np.log(DV_results_matrix[:, 3]), label='DV - $\psi_B$', linewidth=5,
            linestyle='--')

    ax.plot(np.linspace(0, N * 0.05, N), np.log(CCA_results_matrix[:, 2]), label='CCA - $\psi_A$', linewidth=5)
    ax.plot(np.linspace(0, N * 0.05, N), np.log(CCA_results_matrix[:, 3]), label='CCA - $\psi_B$', linewidth=5,
            linestyle='--')

    ax.plot(np.linspace(0, N * 0.05, N), np.log(FKT_results_matrix[:, 2]), label='FKT - $\psi_A$', linewidth=5)
    ax.plot(np.linspace(0, N * 0.05, N), np.log(FKT_results_matrix[:, 3]), label='FKT - $\psi_B$', linewidth=5,
            linestyle='--')

    ax.set_ylabel("log(SNR)", fontsize=40, **csfont)
    ax.set_xlabel("$\sigma$", fontsize=40, **csfont)
    ax.set_title("(D)", fontsize=42, **csfont, pad=20)
    plt.xticks(np.round(np.linspace(0, N * 0.05, 5), 2), fontsize=38, rotation=0, **csfont)
    plt.yticks(fontsize=38, rotation=0, **csfont)
    plt.legend(loc="upper right", prop=font)

    plt.subplots_adjust(wspace=0.25)
    plt.tight_layout()
    # center text
    # fig.text(.5, -.25, txt, ha='center', fontsize = 16)
    plt.show()


# Load dataset (SS1 - video of rotating Yoda and bulldog. SS2 - video of rotating bunny and bulldog)
SS1 = sio.loadmat('ss1.mat')
SS2 = sio.loadmat('ss2.mat')
X_A = SS1['ss1']
X_B = SS2['ss2']

# Find the ground truth vectors for the rotation angle:
X1 = SS1['ss1'].reshape((X_A.shape[0],80,60))
X2 = SS2['ss2'].reshape((X_B.shape[0],80,60))

# First, we cut the videos so they will contain only the desired figure:
X1_Y = X1[:,:40,:] # The Yoda
X2_R = X2[:,40:,:] # The bunny (Rabbit)
# reshape to 2D array
X1_Y = X1_Y.reshape(X1_Y.shape[0],40*60)
X2_R = X2_R.reshape(X2_R.shape[0],40*60)

# compute the kernel, Laplacian and leading eigenvector of Yoda
KY = Kernel_matrix(X1_Y,300)
LY,dY,vY = LG_sym(KY)

# compute the kernel, Laplacian and leading eigenvector of bunny
KR = Kernel_matrix(X2_R,300)
LR,dR,vR = LG_sym(KR)

# Run all three methods N = 20 times:
DV_results_matrix, CCA_results_matrix, FKT_results_matrix = simulation(X_A, X_B,vY[:,1],vR[:,1], N= 20)
#Plot results:
plot_results(DV_results_matrix, CCA_results_matrix,FKT_results_matrix,N=20)