import tensorflow as tf

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
def make_unet_model():

    # noise is already added to the data
    # length dimension is not fixed
    data = tf.keras.Input(shape=(None,6))

    mask_lstm = tf.cast( data[:,:,-1], bool )
    mask = data[:,:,-1:]

    # pre-analyze wtih rnn
    lstm_layer_pre = tf.keras.layers.LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)
    bidir_pre = tf.keras.layers.Bidirectional( lstm_layer_pre, merge_mode='mul' )
    
    x = bidir_pre(data, mask=mask_lstm)
    x = x*mask

    enc_filters = [80, 96, 48]
    enc_kernels = [12, 10, 8]
    enc_regs = [0.0, 0.0, 0.0]

    encoder = make_encoder_cnn(None,64,3,enc_filters,enc_kernels,enc_regs)
    x = tf.concat((x,mask), axis=-1)
    encs = encoder(x)

    dec_filters = [96, 112, 96]
    dec_kernels = [10, 12, 14]
    dec_regs = [0.0, 0.0, 0.0]

    rev = list(reversed(encs))
    shapes = [ r.shape[1:] for r in rev ]

    decoder = make_decoder_cnn(3,dec_filters,dec_kernels,dec_regs,shapes)
    x = decoder(rev)
    
    # post-rnn
    lstm_layer = tf.keras.layers.LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)
    bidir = tf.keras.layers.Bidirectional( lstm_layer, merge_mode='mul' )
    
    x = bidir(x, mask=mask_lstm)
    x = x*mask

    x = tf.keras.layers.Conv1D(2, 4, padding='same')(x)
    
    # assign auxiliary hits to the noise class
    preds = tf.where( tf.cast(mask,bool), tf.keras.layers.Softmax(axis=-1)(x), tf.constant([0.,1.]) )

    model = tf.keras.Model( inputs=data, outputs=preds )
    return model
