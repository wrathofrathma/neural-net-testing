{
    "decay_epochs": 10,
    "decay_rate": 1e-05,
    "error": "mse",
    "input_size": 3,
    "layers": [
        {
            "_layer_num": 0,
            "activation": "tanh",
            "biases": [
                0.0,
                0.0,
                0.0
            ],
            "n_nodes": 3,
            "weights": [
                [
                    1,
                    1,
                    1
                ],
                [
		    2,2,2
                ],
                [
3,3,3
                ]
            ]
        },
        {
            "_layer_num": 1,
            "activation": "tanh",
            "biases": [
                0.0,
                0.0,
                0.0
            ],
            "n_nodes": 3,
            "weights": [
                [
1,2,3
                ],
                [
1,2,3
                ],
                [
1,2,3
                ]
            ]
        },
        {
            "_layer_num": 2,
            "activation": "tanh",
            "biases": [
                0.0,
                0.0,
                0.0
            ],
            "n_nodes": 3,
            "weights": [
                [
0.5
                ],
                [
0.5
                ],
                [
0.5
                ]
            ]
        },
        {
            "_layer_num": 3,
            "activation": "tanh",
            "biases": [
                0.0
            ],
            "n_nodes": 1,
            "weights": [
                [
                    0.5
                ]
            ]
        }
    ],
    "learn_rate": 0.001,
    "loss": "placeholder",
    "output_size": 1
}
