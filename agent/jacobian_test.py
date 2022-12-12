
def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])

def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    '''
        Make params regular Tensors instead of nn.Parameter
    '''
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names

def _set_nested_attr(obj: Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)

def load_weights(mod: Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)

def compute_jacobian(model, obs, acs):
    '''

    @param model: model with vector output (not scalar output!) the parameters of which we want to compute the Jacobian for
    @param x: input since any gradients requires some input
    @return: either store jac directly in parameters or store them differently

    we'll be working on a copy of the model because we don't want to interfere with the optimizers and other functionality
    '''

    jac_model = copy.deepcopy(model) # because we're messing around with parameters (deleting, reinstating etc)
    all_params, all_names = extract_weights(jac_model) # "deparameterize weights"
    load_weights(jac_model, all_names, all_params) # reinstate all weights as plain tensors

    def param_as_input_func(model, obs_batch, acs_batch, param):
        load_weights(model, [name], [param]) # name is from the outer scope
        out = model(obs_batch, acs_batch).flatten()
        return out

    jac_list = []

    for i, (name, param) in enumerate(zip(all_names, all_params)):
        jac = torch.autograd.functional.jacobian(lambda param: param_as_input_func(jac_model, obs, acs, param),
                                                 param,
                                                 strict=True if i==0 else False,
                                                 vectorize=False if i==0 else True)
        jac_list.append(jac)

    return jac_list
    #del jac_model # cleaning up




    def get_param_grad(self, obs, acs):
        assert obs.shape[0] == acs.shape[0]
        batch = obs.shape[0]

        # copy the model in case autograd.jacobian does some form of in place operations
        model_copy = copy.deepcopy(self)
        model_copy.zero_grad()

        # Get current parameters
        # parameter_to_vector returns a shallow copy of the parameters, so changes
        # will affect the original parameters
        cur_params = nn.utils.parameters_to_vector(model_copy.parameters()).detach()
        cur_params.requires_grad_()

        def load_weights_then_forward(model, param_vec, batch_obs, batch_acs):
            # parameters.data now points to param_vec,
            # i.e changes to param_vec will change parameters.data
            vector_to_parameters(param_vec, model.parameters())

            import pdb
            pdb.set_trace()

            return model.forward(batch_obs, batch_acs).flatten()

        param_func = lambda p_vec: load_weights_then_forward(model_copy, p_vec, obs, acs)
#        param_func = lambda p_vec: p_vec
#        out = param_func(cur_params)
#        import pdb
#        pdb.set_trace()
#

        # Parameter gradient should have size [batch, param_dim]
        param_grad = torch.autograd.functional.jacobian(param_func,
                                                        cur_params,
                                                        strict=False,
                                                        vectorize=False)

#        assert param_grad.shape[0] == batch, param_grad.shape[1] == self.param_dim

        return param_grad
