import tensorflow as tf

# Multiplicative noise layer class
class MultNoise(tf.keras.layers.Layer):
    def __init__(self):
        super(MultNoise, self).__init__()

def call(self, inputs, training=False):
    # Qs - integral charge, mask - mask of auxiliary hits
    Qs = inputs[:,:,:1]
    mask = inputs[:,:,-1:]
    if training:
        Qs_n_mult = tf.random.normal(Qs.shape, mean=Q_mean_noise, stddev=Q_noise_fraction*Qs)*mask
        Qs_n = Qs + Qs_n_mult
    else:
        Qs_n = Qs
    return Qs_n

### U-net like model

# encoder, downsamples 2x n times
def make_encoder_cnn(init_size,init_channels,n,filters,kernels,regularizations):

    data_in = tf.keras.Input(shape=(init_size,init_channels+1))

    data = data_in[:,:,:-1]
    mask = data_in[:,:,-1:]

    # encs stores data to be concatanated in the decoder (upsampling) block
    encs = [data_in]
    
    assert len(filters)==n
    assert len(filters)==len(kernels)
    assert len(filters)==len(regularizations)

    x = data
    # downsample 2x n times
    for (fil,ker,reg) in zip(filters,kernels,regularizations):
        x = tf.keras.layers.Conv1D(fil, ker, padding='same')(x)
        x = tf.keras.layers.PReLU ( shared_axes=[1], alpha_regularizer=tf.keras.regularizers.L2(reg) )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # multiply with mask to put auxiliary hits to zero
        x = x*mask
        # skip for residual connection within the block
        skip = x
        x = tf.keras.layers.Conv1D(fil, ker, padding='same')(x)
        x = tf.keras.layers.PReLU ( shared_axes=[1], alpha_regularizer=tf.keras.regularizers.L2(reg) )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = x*mask
        x = tf.concat((x,skip), axis=-1)
        x = tf.keras.layers.Conv1D(fil, ker, strides=2, padding='same')(x)
        x = tf.keras.layers.PReLU ( shared_axes=[1], alpha_regularizer=tf.keras.regularizers.L2(reg) )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # get mask for the reduced data
        mask = tf.keras.layers.MaxPool1D( pool_size=2, padding='same' )(mask)
        dat = tf.concat((x,mask), axis=-1)
        encs.append(dat)

    model = keras.Model( inputs=data_in, outputs=encs )
    return model

# decoder, upsamples 2x n times
def make_decoder_cnn(n,filters,kernels,regularizations,shapes):

    # data from the downsampling (encoder) block
    skips = [ tf.keras.Input(shape=shape) for shape in shapes ]
    
    assert len(filters)==n
    assert len(filters)==len(kernels)
    assert len(filters)==len(regularizations)
    assert len(filters)==len(shapes)-1

    x = skips[0][:,:,:-1]
    mask = skips[0][:,:,-1:]

    # upsample 2x n times
    for (fil,ker,skip,reg) in zip(filters,kernels,skips[1:],regularizations):
        x = tf.keras.layers.Conv1D(fil, ker, padding='same')(x)
        x = tf.keras.layers.PReLU ( shared_axes=[1], alpha_regularizer=tf.keras.regularizers.L2(reg) )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = x*mask
        l_skip = x
        x = tf.keras.layers.Conv1D(fil, ker, padding='same')(x)
        x = tf.keras.layers.PReLU ( shared_axes=[1], alpha_regularizer=tf.keras.regularizers.L2(reg) )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = x*mask
        x = tf.concat((x,l_skip), axis=-1)
        x = tf.keras.layers.Conv1DTranspose(fil, ker, strides=2, padding='same', output_padding=None)(x)
        x = tf.keras.layers.PReLU ( shared_axes=[1], alpha_regularizer=tf.keras.regularizers.L2(reg) )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # make data of the proper `length' (same as in the encoder blocks) 
        z = x[:,:tf.shape(skip)[1],:]
        mask = skip[:,:,-1:]
        z = z*mask
        x = tf.concat( (z,skip[:,:,:-1]), axis=-1 )
        
    decs = x

    model = keras.Model( inputs=skips, outputs=x )
    return model

# make u-net model
def make_unet_model(stds_gauss, Q_mean_noise):

    data = tf.keras.Input(shape=(None,6))

    # split channels to add noise at different rates
    Qs = data[:,:,:1]
    Ts = data[:,:,1:2]
    Xs = data[:,:,2:3]
    Ys = data[:,:,3:4]
    Zs = data[:,:,4:5]
    mask = data[:,:,-1:]

    # addative noise
    Qs_n = tf.keras.layers.GaussianNoise(stds_gauss[0])(Qs)
    Ts_n = tf.keras.layers.GaussianNoise(stds_gauss[1])(Ts)
    Xs_n = tf.keras.layers.GaussianNoise(stds_gauss[2])(Xs)
    Ys_n = tf.keras.layers.GaussianNoise(stds_gauss[3])(Ys)
    Zs_n = tf.keras.layers.GaussianNoise(stds_gauss[4])(Zs)
    
    # multplicative noise
    Qs_in = tf.concat((Qs_n,mask), axis=-1)
    Qs_n = MultNoise()(Qs_in)

    # reoder according to activation times after adding noise
    sort_idxs = tf.argsort( Ts_n, axis=-1 )
    Qs_n = tf.gather( Qs_n, sort_idxs, batch_dims=-1 )
    Ts_n = tf.gather( Ts_n, sort_idxs, batch_dims=-1 )
    Xs_n = tf.gather( Xs_n, sort_idxs, batch_dims=-1 )
    Ys_n = tf.gather( Ys_n, sort_idxs, batch_dims=-1 )
    Zs_n = tf.gather( Zs_n, sort_idxs, batch_dims=-1 )
    mask = tf.gather( mask, sort_idxs, batch_dims=-1 )

    mask_lstm = tf.cast( tf.squeeze(mask, axis=-1), bool )
    noise_data = tf.concat((Qs_n,Ts_n,Xs_n,Ys_n,Zs_n), axis=-1)

    # pre-analyze wtih rnn
    lstm_layer_pre = tf.keras.layers.LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)
    bidir_pre = tf.keras.layers.Bidirectional( lstm_layer_pre, merge_mode='mul' )
    
    x = bidir_pre(noise_data, mask=mask_lstm)

    enc_filters = [80, 96, 48]
    enc_kernels = [12, 10, 8]
    enc_regs = [0.0, 0.0, 0.0]

    encoder = make_encoder_cnn(None,64,3,enc_filters,enc_kernels,enc_regs)
    x = tf.concat((x,mask), axis=-1)
    encs = encoder(x)

    dec_filters = [96, 112, 96]
    dec_kernels = [10, 12, 14]
    dec_regs = [0.0, 0.0, 0.0]

    # reverse the order of encs and get their shapes to make decoder (upsampling) block
    rev = list(reversed(encs))
    shapes = [ r.shape[1:] for r in rev ]

    decoder = make_decoder_cnn(3,dec_filters,dec_kernels,dec_regs,shapes)
    x = decoder(rev)
    
    # post-rnn
    lstm_layer = tf.keras.layers.LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)
    bidir = tf.keras.layers.Bidirectional( lstm_layer, merge_mode='mul' )
    
    x = bidir(x, mask=mask_lstm)

    x = tf.keras.layers.Conv1D(2, 4, padding='same')(x)
    
    # make auxiliary hits to always belong to the noise
    preds = tf.where( tf.cast(mask,bool), tf.keras.layers.Softmax(axis=-1)(x), tf.constant([0.,1.]) )

    model = keras.Model( inputs=data, outputs=preds )
    return model