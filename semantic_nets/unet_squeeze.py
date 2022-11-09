from keras.models import *
from keras.layers import *
from semantic_nets.mobilenet_squeeze import get_mobilenet_encoder


IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1
def relu6(x):
    # relu函数
    return K.relu(x, max_value=6.0)

def hard_swish(x):
    # 利用relu函数乘上x模拟sigmoid
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

def return_activation(x, nl):
    # 用于判断使用哪个激活函数
    if nl == 'HS':
        x = Activation(hard_swish)(x)
    if nl == 'RE':
        x = Activation(relu6)(x)

    return x

def squeeze(inputs):
    # 注意力机制单元
    input_channels = int(inputs.shape[-1])
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(int(input_channels/4))(x)
    x = Activation(relu6)(x)
    x = Dense(input_channels)(x)
    x = Activation(hard_swish)(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])
    return x

def _unet( n_classes , encoder , l1_skip_conn=True,  input_height=416, input_width=608  ):

	img_input , levels = encoder( input_height=input_height ,  input_width=input_width )
	[f1 , f2 , f3 , f4 , f5 ] = levels 

	o = f4
	# 26,26,512
	o = ( ZeroPadding2D( (1,1) , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	# 52,52,512
	o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	# 52,52,768，concatenate
	o = ( concatenate([ o ,f3],axis=MERGE_AXIS )  )
	# 注意力模块
	o = squeeze(o)
	o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
	# 52,52,256
	o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	# 104,104,256
	o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	# 104,104,384，concatenate
	o = ( concatenate([o,f2],axis=MERGE_AXIS ) )
	# 注意力模块
	o = squeeze(o)
	o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING ))(o)
	# 104,104,128
	o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format=IMAGE_ORDERING ) )(o)
	o = ( BatchNormalization())(o)
	# 208,208,128
	o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	
	if l1_skip_conn:
		o = ( concatenate([o,f1],axis=MERGE_AXIS ) )

	o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING ))(o)
	o = ( BatchNormalization())(o)

	o =  Conv2D( n_classes , (3, 3) , padding='same', data_format=IMAGE_ORDERING )( o )
	
	# 将结果进行reshape
	o = Reshape((int(input_height/2)*int(input_width/2), -1))(o)
	o = Softmax()(o)
	model = Model(img_input,o)

	return model



def mobilenet_unet( n_classes ,  input_height=224, input_width=224 , encoder_level=3):

	model =  _unet( n_classes , get_mobilenet_encoder ,  input_height=input_height, input_width=input_width  )
	model.model_name = "mobilenet_unet"
	return model


