#File che contiene le matrici necessarie a calcolare la norme FE e FV

import numpy

def set_coef12():
    coef = numpy.array([0.0254224531851030,   0.0254224531851030,   0.0254224531851030,   0.0583931378631890,   0.0583931378631890,   0.0583931378631890,  0.0414255378091870,   0.0414255378091870,   0.0414255378091870,   0.0414255378091870,   0.0414255378091870,   0.0414255378091870])

    return coef

def set_coor12():

    coor = numpy.array([[0.0630890144915020,   0.0630890144915020],
                        [0.8738219710169960,   0.0630890144915020],
                        [0.0630890144915020,   0.8738219710169960],
                        [0.2492867451709100,   0.2492867451709100],
                        [0.5014265096581800,   0.2492867451709100],
                        [0.2492867451709100,   0.5014265096581800],
                        [0.0531450498448160,   0.3103524510337850],
                        [0.6365024991213990,   0.0531450498448160],
                        [0.0531450498448160,   0.6365024991213990],
                        [0.3103524510337850,   0.0531450498448160],
                        [0.6365024991213990,   0.3103524510337850],
                        [0.3103524510337850,   0.6365024991213990]])
    return coor

def set_base1_12():

    base1 = numpy.array([[0.8738219710169961,   0.0630890144915020,   0.0630890144915020],
                         [0.0630890144915020,   0.8738219710169960,   0.0630890144915020],
                         [0.0630890144915021,   0.0630890144915020,   0.8738219710169960],
                         [0.5014265096581800,   0.2492867451709100,   0.2492867451709100],
                         [0.2492867451709100,   0.5014265096581800,   0.2492867451709100],
                         [0.2492867451709100,   0.2492867451709100,   0.5014265096581800],
                         [0.6365024991213990,   0.0531450498448160,   0.3103524510337850],
                         [0.3103524510337850,   0.6365024991213990,   0.0531450498448160],
                         [0.3103524510337849,   0.0531450498448160,   0.6365024991213990],
                         [0.6365024991213990,   0.3103524510337850,   0.0531450498448160],
                         [0.0531450498448160,   0.6365024991213990,   0.3103524510337850],
                         [0.0531450498448161,   0.3103524510337850,   0.6365024991213990]])
    return base1

def set_base2_12():

    base2 = numpy.array([[ 0.65330770304705954,  -0.05512856699248411,  -0.05512856699248411,   0.01592089499803580,   0.22051426796993640,   0.22051426796993640],
                         [-0.05512856699248405,   0.65330770304705965,  -0.05512856699248411,   0.22051426796993642,   0.01592089499803579,   0.22051426796993612],
                         [-0.05512856699248414,  -0.05512856699248411,   0.65330770304705965,   0.22051426796993642,   0.22051426796993612,   0.01592089499803579],
                         [ 0.00143057951778969,  -0.12499898253509756,  -0.12499898253509756,   0.24857552527162491,   0.49999593014039018,   0.49999593014039018],
                         [-0.12499898253509753,   0.00143057951778980,  -0.12499898253509756,   0.49999593014039029,   0.24857552527162485,   0.49999593014039023],
                         [-0.12499898253509756,  -0.12499898253509756,   0.00143057951778980,   0.49999593014039029,   0.49999593014039023,   0.24857552527162485],
                         [ 0.17376836365417397,  -0.04749625719880005,  -0.11771516330842915,   0.06597478591860528,   0.79016044276582309,   0.13530782816862683],
                         [-0.11771516330842910,   0.17376836365417403,  -0.04749625719880005,   0.13530782816862683,   0.06597478591860527,   0.79016044276582331],
                         [-0.11771516330842935,  -0.04749625719880005,   0.17376836365417403,   0.13530782816862683,   0.79016044276582331,   0.06597478591860527],
                         [ 0.17376836365417400,  -0.11771516330842915,  -0.04749625719880005,   0.06597478591860528,   0.13530782816862683,   0.79016044276582309],
                         [-0.04749625719880010,   0.17376836365417403,  -0.11771516330842915,   0.79016044276582309,   0.06597478591860523,   0.13530782816862685],
                         [-0.04749625719880013,  -0.11771516330842915,   0.17376836365417403,   0.79016044276582309,   0.13530782816862685,   0.06597478591860523]])
    return base2

def set_coef4():

    coef = numpy.array([-0.281250000000000,   0.260416666666667,   0.260416666666667,   0.260416666666667])

    return coef4

def set_coor4():

    coor = numpy.array([[0.333333333333333,   0.333333333333333],
                        [0.200000000000000,   0.200000000000000],
                        [0.600000000000000,   0.200000000000000],
                        [0.200000000000000,   0.600000000000000]])
    return coor

def set_base1_4():

    base1 = numpy.array([[0.333333333333333,   0.333333333333333,   0.333333333333333],
                         [0.600000000000000,   0.200000000000000,   0.200000000000000],
                         [0.200000000000000,   0.600000000000000,   0.200000000000000],
                         [0.200000000000000,   0.200000000000000,   0.600000000000000]])

    return base1

def set_base2_4():

    base2 = numpy.array([[-0.111111111111111,  -0.111111111111111,  -0.111111111111111,   0.444444444444444,   0.444444444444444,   0.444444444444444],
                         [ 0.120000000000000,  -0.120000000000000,  -0.120000000000000,   0.160000000000000,   0.480000000000000,   0.480000000000000],
                         [-0.120000000000000,   0.120000000000000,  -0.120000000000000,   0.480000000000000,   0.160000000000000,   0.480000000000000],
                         [-0.120000000000000,  -0.120000000000000,   0.120000000000000,   0.480000000000000,   0.480000000000000,   0.160000000000000]])

    return base_2
