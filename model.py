from keras.layers import Input, BatchNormalization, Convolution2D, Deconvolution2D, Activation, Merge, advanced_activations
from keras.models import Sequential

class Generative:
    def __init__(self, shape):
        width = img_width
        height = img_height
        model = Sequential()

    def gen(self, input):
        x = Convolution2D(64, 3, 3, activation='relu', name='sr_gen_conv_1')(input)

        for i in range(6):
            x = self.B_res(x, i + 1)

        x = Deconvolution2D(64, 3, 3, name='sr_gen_deconv_1')(x)
        x = Deconvolution2D(64, 3, 3, name='sr_gen_deconv_2')(x)
        x = Convolution2D(64, 3, 3, name='sr_gen_conv_2')(x)

        return x

    def B_res(self, input, id):
        init = input
        x = Convolution2D(64, 3, 3, name='sr_gen_res_conv_' + str(id) + '_1')(input)
        x = BatchNormalization(name='sr_res_batchnorm_' + str(id) + '_1', activation='relu')(x)

        x = Convolution2D(64, 3, 3, name='sr_gen_res_conv_' + str(id) + '_2')(input)
        x = BatchNormalization(name='sr_res_batchnorm_' + str(id) + '_2', activation='relu')(x)

        m = merge([x, init], mode='sum', name='sr_gen_res_merge_' + str(id))

        return m

class Discriminator:
    def __init__(self, img_width=384, img_height=384, adversarial_loss_weight=1e-3, small_model=False):
        self.img_width = img_width
        self.img_height = img_height
        self.adversarial_loss_weight = adversarial_loss_weight
        self.mode = 2
        self.gan_layers = None

    def append_gan_network(self, x_in, true_X_input):

        # Append the inputs to the output of the SRResNet
        x = merge([x_in, true_X_input], mode='concat', concat_axis=0)

        # Normalize the inputs via custom VGG Normalization layer
        x = Normalize(type="gan", value=255., name="gan_normalize")(x)

        x = Convolution2D(64, self.k, self.k, border_mode='same', name='gan_conv1_1')(x)
        x = LeakyReLU(0.3, name="gan_lrelu1_1")(x)

        x = Convolution2D(64, self.k, self.k, border_mode='same', name='gan_conv1_2', subsample=(2, 2))(x)
        x = LeakyReLU(0.3, name='gan_lrelu1_2')(x)
        x = BatchNormalization(mode=self.mode, axis=1, name='gan_batchnorm1_1')(x)

        filters = [128, 256]

        for i, nb_filters in enumerate(filters):
            for j in range(2):
                subsample = (2, 2) if j == 1 else (1, 1)

                x = Convolution2D(nb_filters, self.k, self.k, border_mode='same', subsample=subsample,
                                  name='gan_conv%d_%d' % (i + 2, j + 1))(x)
                x = LeakyReLU(0.3, name='gan_lrelu_%d_%d' % (i + 2, j + 1))(x)
                x = BatchNormalization(mode=self.mode, axis=1, name='gan_batchnorm%d_%d' % (i + 2, j + 1))(x)

        x = Flatten(name='gan_flatten')(x)

        output_dim = 128 if self.small_model else 1024

        x = Dense(output_dim, name='gan_dense1')(x)
        x = LeakyReLU(0.3, name='gan_lrelu5')(x)

        gan_regulrizer = AdversarialLossRegularizer(weight=self.adversarial_loss_weight)
        x = Dense(1, activation="sigmoid", activity_regularizer=gan_regulrizer, name='gan_output')(x)

        return x
