import os
import re

save_weights=False

if save_weights:
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    #export TF_USE_LEGACY_KERAS=1
    import glob
    import re
    import tensorflow as tf
    mdlfns=glob.glob("/home/ubuntu/.antspymm/siq*mdl.h5")
    for k in range( len( mdlfns ) ):
        wtnm=re.sub( "mdl.h5", "weights.h5", mdlfns[k] )
        print( wtnm )
        mdl = tf.keras.models.load_model( os.path.expanduser( mdlfns[k] ), compile=False )
        mdl.save_weights( wtnm )
else:  # load weights and save models
    os.environ['TF_USE_LEGACY_KERAS'] = '0'
    import glob
    import re
    import tensorflow as tf
    mdlfns=glob.glob("/home/ubuntu/.antspymm/siq*mdl.h5")
    for k in range( len( mdlfns ) ):
        k3nm=re.sub( "mdl.h5", "mdlk3.h5", mdlfns[k] )
        print( k3nm )
        # Define regex patterns
        option_pattern = r'siq_(\w+?)short'
        upper_pattern = r'(\d+)x(\d+)x(\d+)'
        nchan_pattern = r'_(\d+)chan'
        # Parse the 'option'
        option=None
        option_match = re.search(option_pattern, k3nm)
        if option_match:
            option = option_match.group(1)
        # Parse the 'upper'
        upper_match = re.search(upper_pattern, k3nm)
        upper=None
        if upper_match:
            upper = [int(upper_match.group(1)), int(upper_match.group(2)), int(upper_match.group(3))]
        # Parse the 'nchan'
        nchan_match = re.search(nchan_pattern, k3nm)
        nchan=None
        if nchan_match:
            nchan = int(nchan_match.group(1))
        # NOTE: have not figured out 2 chan translation yet
        if nchan == 1 and os.path.exists( k3nm ):
            # Print the results
            print("option =", option)  # Output: 'small'
            print("upper =", upper)    # Output: [2, 2, 2]
            print("nchan =", nchan)    # Output: 1
            if option is not None:
                mdl = siq.default_dbpn( upper, option=option, nChannelsIn=nchan, nChannelsOut=nchan, sigmoid_second_channel=nchan==2 )
                mdl.load_weights( k3nm )
            else:
                mdl = siq.default_dbpn( upper, option=option, nChannelsIn=nchan, nChannelsOut=nchan, sigmoid_second_channel=nchan==2 )
                mdl.load_weights( k3nm )
     
