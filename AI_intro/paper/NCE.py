import numpy as np

def NCE(source_label: np.ndarray, target_label: np.ndarray):
    C_t = int(np.amax(target_label) + 1)
    C_s = int(np.amax(source_label) + 1)
    N = len(source_label)

    joint = np.zeros((C_t, C_s), dtype=float)   #P(z,y), [C_t, C_s]
    for s, t in zip(source_label, target_label):
        s = int(s)
        t = int(t)
        joint[t, s] += 1.0 / N

    p_z = joint.sum(axis=0, keepdims=True)  #P(z), [1, C_s]
    p_target_given_source = (joint / p_z).T #P(y|z), [C_s, C_t]
    mask = p_z.reshape(-1) != 0 #过滤 P(z)==0
    p_target_given_source = p_target_given_source[mask] + 1e-20  #防止 log(0)
    entropy_y_given_z = np.sum(- p_target_given_source * np.log(p_target_given_source), axis=1, keepdims=True)  #H(y|z)
    conditional_entropy = np.inner(entropy_y_given_z, p_z.reshape((-1, 1))[mask])    #CE
    return - conditional_entropy    #NCE
