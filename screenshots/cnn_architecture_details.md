# CNN Architecture Details (model.h5)

- Total parameters: **11,963,587**

| # | Layer (name) | Type | Output shape | Params | Hyperparameters |
|---:|---|---|---|---:|---|
| 1 | `conv2d` | `Conv2D` | `(?, 223, 223, 32)` | 896 | filters=32, kernel=(3, 3), strides=(1, 1), padding=valid, act=relu |
| 2 | `max_pooling2d` | `MaxPooling2D` | `(?, 111, 111, 32)` | 0 | pool=(2, 2), strides=(2, 2), padding=valid |
| 3 | `conv2d_1` | `Conv2D` | `(?, 109, 109, 64)` | 18,496 | filters=64, kernel=(3, 3), strides=(1, 1), padding=valid, act=relu |
| 4 | `max_pooling2d_1` | `MaxPooling2D` | `(?, 54, 54, 64)` | 0 | pool=(2, 2), strides=(2, 2), padding=valid |
| 5 | `flatten` | `Flatten` | `(?, 186624)` | 0 |  |
| 6 | `dense` | `Dense` | `(?, 64)` | 11,944,000 | units=64, act=relu |
| 7 | `dense_1` | `Dense` | `(?, 3)` | 195 | units=3, act=softmax |
