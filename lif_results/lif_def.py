lif_neuron = UserNeuron(vars={"v": ("Isyn - v/((max_tau / (1 + e ** (-tau))) + eps)", "rst")},
                        threshold="v - v_thr",
                        output_var_name="v",
                        param_vals={"tau": tau, "rst": 0, "v_thr": 1, "max_tau":20, "eps":1e-6, "e": np.exp(1)},
                        var_vals={"v": 0})
