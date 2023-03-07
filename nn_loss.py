import tensorflow as tf

# loss that penalizes NN for predictiong hits with big time residuals as signal hits
class tres_penalty(tf.keras.losses.Loss):

    def __init__(self, pen_coeff, t_res_lim, mark_big_tres_as_noise, loss_norm_coeff):
        super().__init__()
        # coefficient for the sum with entropy loss
        self.pen_coeff = pen_coeff
        # time residual threshold below which NNs is not penalized
        self.t_res_lim = t_res_lim
        # whether to apply penalty for signal hits as well
        self.apply_pen_for_signal = mark_big_tres_as_noise
        # coefficient to renormilize the total loss
        self.loss_norm_coeff = loss_norm_coeff
        
    def __call__(self, y_true, y_pred, sample_weight):
        # first two channels of labels carry one-hot encodings of the correct class
        # 3rd channel is time residual
        label = y_true[:,:,:2]
        t_res = y_true[:,:,2]
        # get the mask of auxiliary hits by tracking corresponding values in tres (predefined to be 1e5)
        mask = tf.where( t_res>=9*1e4, 1., 0. )
        # calculate entropy loss
        entropy = tf.keras.losses.binary_crossentropy( label, y_pred )
        entropy = tf.math.reduce_mean( tf.math.multiply( entropy, sample_weight ) )
        # penalty for false-signal and signal with large tres, proportional to tres and NNs confidence
        t_res = tf.math.abs(t_res)
        sig_confidence = y_pred[:,:,0]
        class_preds = tf.math.argmax(y_pred, axis=-1)
        class_true = tf.math.argmax(label, axis=-1)
        # mask of false-signals
        fs_mask = tf.where( tf.math.logical_and(class_preds==0,class_true==1), 1., 0. )
        # mask of true-signals with tres above threshold
        big_tres_mask = tf.where( tf.math.logical_and( tf.math.logical_and(class_preds==0,t_res>=self.t_res_lim),
                                                      class_true==0), 1., 0. )
        # total mask
        tres_pen_mask = fs_mask+self.apply_pen_for_signal*big_tres_mask
        # calculate penalty for each hit
        tres_pen = self.pen_coeff * sig_confidence*t_res*tres_pen_mask
        tres_pen = tf.math.reduce_mean( tf.math.multiply( tres_pen, sample_weight ) )
        # calculate total loss
        loss = self.loss_norm_coeff*(tres_pen+entropy)
        return loss