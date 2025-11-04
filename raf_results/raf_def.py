raf_neuron = UserNeuron(vars={"x": ("Isyn + ((max_b / (1 + e ** (-b))) + min_b) * x - ((max_w / (1 + e ** (-w))) + min_w) * y", "x"), "y": ("((max_w / (1 + e ** (-w))) + min_w) * x + ((max_b / (1 + e ** (-b))) - min_b) * y", "y")},
                        threshold="y - a_thresh",
                        output_var_name="y",
                        param_vals={"b": b_raf, "w": w_raf, "a_thresh": 1, "max_b": -1.0, "min_b": -0.0007, "max_w": max_w 0.1, "min_w": 0.0045, "e": np.exp(1)},
                        var_vals={"x": 0.0, "y": 0.0, "q": 0.0},
                        sub_steps=100,
                        solver="linear_euler"
)
