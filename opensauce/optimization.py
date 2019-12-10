import numpy as np
import scipy.optimize as scio

def fminsearchbnd(fxn, x0, LB=None, UB=None, options=None):
    #fminsearch, but with bound constraints by transformation
    exitflag = 0
    output = 0
    xsize = len(x0)
    x0 = x0[:]
    n = len(x0)

    if LB == None:
        LB = np.matlib.repmat(-1* np.inf, n, 1)
    else:
        LB = LB[:]

    if UB == None:
        UB = np.matlib.repmat(np.inf, n, 1)
    else:
        UB = UB[:]

    assert(len(LB) == n)
    assert(len(UB) == n)

    if options == None:
        options = {"FunValCheck": "off",
            "MaxFunEvals": 400,
            "MaxIter": 400,
            "OutputFcn": [],
            "TolFun": 1.0*(10**(-7)),
            "TolX": 1.0*(10**(-4)) }
    
    params = {
        "LB": LB,
        "UB": UB,
        "fxn": fxn,
        "n": n,
        "OutputFcn": [],
        "BoundClass": np.zeros(n,1)
    }

    for i in range(n):
        k = np.isfinite(LB[i]) + 2*np.isfinite(UB[i])
        params['BoundClass'][i] = k
        if k == 3 and LB[i] == UB[i]:
            params['BoundClass'][i] = 4

    # transform starting values into unconstrained surrogates
    # check for infeasbile starting values
    x0u = x0
    k = 0
    for i in range(n):
        if params['BoundClass'][i] == 1:
            #lower bound only
            if x0[i] <= LB[i]:
            #infeasible, use bound
                x0u[k] = 0
            else:
                x0u[k] = np.sqrt(x0[i] - LB[i])
            k += 1
        
        elif params['BoundClass'][i] == 2:
            if x0[i] >= UB[i]:
                x0u[k] = 0
            else:
                x0u[k] = np.sqrt(UB[i] - x0[i])
            k += 1

        elif params['BoundClass'][i] == 3:
            if x0[i] <= LB[i]:
                x0u[k] = -np.pi/2
            elif x0[i] >= UB[i]:
                x0u[k] = np.pi/2
            else:
                x0u[k] = 2*(x0[i] - LB[i])/(UB[i] - LB[i]) - 1
                x0u[k] = 2*np.pi + np.arcsin(max(-1, min(1, x0u[k])))
            k += 1

        elif params['BoundClass'][i] == 0:
            x0u[k] = x0[i]
            k += 1
        
        else:
            pass # don't do anything

    # correct for fixed unknown
    if k <= n:
        x0u[k:n] = []

    if np.size(x0u) == 0:
        # all variables fixed
        x = xtransform(x0u, params)

        x = np.reshape(x, xsize)

        fval = feval(params["fxn"], x) #TODO implement this

        exitflag = 0

        output = {
            "iterations": 0,
            "funcount": 1,
            "algorithm": 'fminsearch',
            "message": "All variables held fixed by applied bounds."
        }

        return x, fval, exitflag, output
    pass
    #TODO outfun_wrapper
    # if np.size(options["OutputFcn"]) > 0:
    #     params["OutputFcn"] = options["OutputFcn"]
    #     options["OutputFcn"] = lambda: outfun_wrapper()

    
    # xu, fval, exitflag, output = np.fmin(lambda: intrafun(), x0u, options, params)

    # xu = xtransform(xu, params)

    # x = np.reshape(x, xsize)

    # def outfun_wrapper(x): #TODO figure out varargin
    #     xtrans = xtransform(x, params)
    #     return params["OutputFcn"](xtrans) # stop


    # return x, fval, exitflag, output


def xtransform(x, params):
    xtrans = np.zeros(1, params["n"])
    k = 1
    for i in range(params["n"]):
        switch = params["BoundClass"]
        if switch == 1:
            xtrans[i] = params["LB"][i] + np.power(x[k], 2)
            k += 1
        elif switch == 2:
            xtrans[i] = params["UB"][i] - np.power(x[k], 2)
            k += 1
        elif switch == 3:
            xtrans[i] = np.sin(x[k] + 1)/2
            xtrans[i] = xtrans[i] * (params["UB"][i] - params["LB"][i]) + params["LB"][i]

            #fix floating point problems
            xtrans[i] = max(params["LB"][i], min(params["UB"][i], xtrans[i]))
            k += 1
        elif switch == 4:
            xtrans[i] = params["LB"][i]
        else:
            xtrans[i] = x[k]
            k += 1

        return xtransform

def feval(funcName, *args):
    return eval(funcName)(*args)

def intrafun(x, params):
    xtrans = xtransform(x, params)
    return feval(params["fxn"], xtrans, params.keys()[:])