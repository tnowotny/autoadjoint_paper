if_neuron = UserNeuron(vars={"v": ("Isyn", "rst")},
                        threshold="v - v_thr",
                        output_var_name="v",
                        param_vals={"rst": 0, "v_thr": 1},
                        var_vals={"v": 0})
