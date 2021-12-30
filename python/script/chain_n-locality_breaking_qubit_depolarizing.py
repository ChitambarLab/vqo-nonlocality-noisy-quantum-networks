from context import QNetOptimizer as QNopt
import pennylane as qml


n = 2

def max_entangled(settings, wires):
	qml.Hadamard(wires=wires[0])
	qml.CNOT(wires=wires[0:2])
	qml.Rot(*settings[0:3], wires=wires[0])


arb_bilocal_prep_nodes = [
	# QNopt.PrepareNode(1,[0,1], max_entangled, 3),
	QNopt.PrepareNode(1,[0,1], qml.templates.subroutines.ArbitraryUnitary, 15),
	# QNopt.PrepareNode(1,[2,3], max_entangled, 3),
	QNopt.PrepareNode(1,[2,3], qml.templates.subroutines.ArbitraryUnitary, 15),
]

def local_rot(settings, wires):
	qml.Rot(*settings[0:3], wires=wires[0])
	qml.Rot(*settings[3:6], wires=wires[1])

arb_bilocal_meas_nodes = [
	QNopt.MeasureNode(2, 2, [0], lambda settings, wires: qml.Rot(*settings, wires=wires), 3),
	QNopt.MeasureNode(2, 2, [1,2], local_rot, 6),
	QNopt.MeasureNode(2, 2, [3], lambda settings, wires: qml.Rot(*settings, wires=wires), 3),
]

noise_nodes = [
	QNopt.NoiseNode([0], lambda settings, wires: qml.AmplitudeDamping(0.5, wires=wires[0])),
	QNopt.NoiseNode([3], lambda settings, wires: qml.AmplitudeDamping(0.5, wires=wires[0]))
]


# gamma = 0.75 maximally entangled states are broken by amplitude damping noise significantly sooner than arbitrary
# gamma 0.5 and 0.45 was able to violate with arbitrary states
bilocal_ansatz = QNopt.NetworkAnsatz(arb_bilocal_prep_nodes, arb_bilocal_meas_nodes, noise_nodes)

bilocal_cost = QNopt.nlocal_chain_cost_22(bilocal_ansatz)

opt_dict = QNopt.gradient_descent(
	bilocal_cost,
	bilocal_ansatz.rand_scenario_settings(),
	num_steps = 100,
	step_size = 0.5,
	sample_width = 5
)