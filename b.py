decay = 0.95
TAU = 0.001


class batch_norm:
    def __init__(self):
        # Define BatchNormalization layer
        self.batch_norm_layer = tf.keras.layers.BatchNormalization(momentum=decay, 
                                                                   epsilon=1e-7, 
                                                                   center=True, 
                                                                   scale=True)

        # Handling target parameters for updates (soft update with TAU)
        self.parForTarget = None

    def __call__(self, inputs, is_training):
        # Apply batch normalization conditionally
        return self.batch_norm_layer(inputs, training=is_training)

    # Set the target for soft update (TAU-based soft update mechanism)
    def set_target(self, parForTarget):
        self.parForTarget = parForTarget
        self.update_scale = self.batch_norm_layer.gamma.assign(
            self.batch_norm_layer.gamma * (1 - TAU) + parForTarget.batch_norm_layer.gamma * TAU
        )
        self.update_beta = self.batch_norm_layer.beta.assign(
            self.batch_norm_layer.beta * (1 - TAU) + parForTarget.batch_norm_layer.beta * TAU
        )
        # Combine the updates into a group
        self.update_target = tf.group(self.update_scale, self.update_beta)

    def update_target_network(self):
        # Optional method to apply the target network update
        if self.parForTarget is not None:
            tf.function(self.update_target)
