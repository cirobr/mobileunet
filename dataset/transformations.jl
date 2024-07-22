# transformations


# tranlational transformations
tr1 = x -> p.flipX(x)               # flipX
tr2 = x -> p.imageRotate(x, π/8)    # rotateR
tr3 = x -> p.imageRotate(x, -π/8)   # rotateL
tr4 = Chain(tr1, tr2)               # flipXrotateR
tr5 = Chain(tr1, tr3)               # flipXrotateL

translational_transformations = [tr1, tr2, tr3, tr4, tr5]
translational_folders = ["flipX/", "rotateR/", "rotateL/", "flipXrotateR/", "flipXrotateL/"]


# pixel transformations
tn1 = x -> p.transf_log2(x; c = 2.f0)                   # log2
tn2 = x -> p.transf_gamma(x; c = 3.f0, gamma = 3.f0)    # gamma30
tn3 = x -> p.transf_gamma(x; c = 1.f0, gamma = 0.5f0)   # gamma05
tn4 = x -> p.transf_grad(x)                             # grad

pixel_transformations = [tn1, tn2, tn3]                 # , tn4]
pixel_folders = ["log2/", "gamma30/", "gamma05/"]       # , "grad/"]
