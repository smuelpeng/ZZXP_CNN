import json,sys
def op_time(name):
    sum_time=0
    name_time=0
    b=a['traceEvents']
    for line in range(0,len(b),2):
        if b[line]['name']=="nin:"+name:
            name_time+=(b[line+1]['ts']-b[line]['ts'])
    return name_time/50,(b[-1]['ts']-b[0]['ts'])/50
if __name__=="__main__":
    a= json.load(open(sys.argv[1]))
    print "MklConvolution",op_time("MklConvolution")
    print "MklConvolution:MKlconvolution_dnnExecute",op_time("MklConvolution:MKlconvolution_dnnExecute")
    print "Convolution",op_time("Convolution")
    print "Input",op_time("Input")
    print "BatchNorm",op_time("BatchNorm")
    print "Scale",op_time("Scale")
    print "Pooling",op_time("Pooling")
    print "ReLU",op_time("ReLU")

