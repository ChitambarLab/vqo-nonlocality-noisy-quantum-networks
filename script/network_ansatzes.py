import pennylane as qml

from context import QNetOptimizer as QNopt

def local_rot(settings, wires):
    qml.Rot(*settings[0:3], wires=wires[0])
    qml.Rot(*settings[3:6], wires=wires[1])

def nlocal_max_entangled_prep_nodes(n):
    return [
        QNopt.PrepareNode(1, [2 * i, 2 * i + 1], QNopt.max_entangled_state, 3) for i in range(n)
    ]

def chain_local_rot_meas_nodes(n):
    meas_nodes = []
    meas_nodes.append(
        QNopt.MeasureNode(2, 2, [0], lambda settings, wires: qml.Rot(*settings, wires=wires), 3)
    )

    meas_nodes.extend(
        [QNopt.MeasureNode(2, 2, [2 * i + 1, 2 * i + 2], local_rot, 6) for i in range(n - 1)]
    )

    meas_nodes.append(
        QNopt.MeasureNode(
            2, 2, [2 * n - 1], lambda settings, wires: qml.Rot(*settings, wires=wires), 3
        )
    )
    return meas_nodes

def chain_bell_meas_nodes(n):
    meas_nodes = []
    meas_nodes.append(
        QNopt.MeasureNode(2, 2, [0], lambda settings, wires: qml.Rot(*settings, wires=wires), 3)
    )

    meas_nodes.extend(
        [QNopt.MeasureNode(2, 2, [2 * i + 1, 2 * i + 2], qml.adjoint(QNopt.max_entangled_state), 3) for i in range(n - 1)]
    )

    meas_nodes.append(
        QNopt.MeasureNode(
            2, 2, [2 * n - 1], lambda settings, wires: qml.Rot(*settings, wires=wires), 3
        )
    )
    return meas_nodes

def chain_arb_meas_nodes(n):
    meas_nodes = []
    meas_nodes.append(
        QNopt.MeasureNode(2, 2, [0], lambda settings, wires: qml.Rot(*settings, wires=wires), 3)
    )

    meas_nodes.extend(
        [QNopt.MeasureNode(
            2, 2, [2 * i + 1, 2 * i + 2], qml.templates.subroutines.ArbitraryUnitary, 15
        ) for i in range(n - 1)]
    )

    meas_nodes.append(
        QNopt.MeasureNode(
            2, 2, [2 * n - 1], lambda settings, wires: qml.Rot(*settings, wires=wires), 3
        )
    )
    return meas_nodes