from build.lib.lnn import UNKNOWN
from lnn import Model, Not, Variable, Or, TRUE, FALSE, UNKNOWN, UPWARD, plot_loss, plot_params

model = Model()
a_forward = model.add_predicates(1, 'a_forward')
a_backward = model.add_predicates(1, 'a_backward')
o = model.add_predicates(1, 'o')

model.add_facts({
    a_forward.name: {
        '0': TRUE,
        '1': TRUE,
        '2': FALSE,
        '3': FALSE,
        '4': FALSE
    },
    a_backward.name: {
        '0': FALSE,
        '1': FALSE,
        '2': TRUE,
        '3': TRUE,
        '4': FALSE
    },
    o.name: {
        '0': TRUE,
        '1': TRUE,
        '2': TRUE,
        '3': TRUE,
        '4': TRUE
    }
})

# x represents the example number
x = Variable('x')
model['valid'] = Or(a_forward(x), a_backward(x), o(x))
model.add_labels({
    'valid': {
        '0': TRUE,
        '1': TRUE,
        '2': FALSE,
        '3': FALSE,
        '4': FALSE
    }
})

parameter_history = {'weights': True, 'bias': True}
losses = ['supervised', 'contradiction']
total_loss, _ = model.train(
    direction=UPWARD, losses=losses, parameter_history=parameter_history)

model.add_facts({
    a_forward.name: {
        't0': TRUE,
        't1': FALSE
    },
    a_backward.name: {
        't0': FALSE,
        't1': TRUE
    },
    o.name: {
        't0': TRUE,
        't1': TRUE
    },
    'valid': {
        't0': UNKNOWN,
        't1': UNKNOWN
    }
})
model.infer(direction=UPWARD)
model['valid'].print(params=True)
# print(model['valid'].get_facts()[-1][-1].item())
# model['valid'].print()
plot_loss(total_loss, losses)
plot_params(model)
