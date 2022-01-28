import pennylane as qml

from context import QNetOptimizer as QNopt


def local_rot(settings, wires):
    for i in range(len(wires)):
        qml.Rot(*settings[3 * i : 3 * i + 3], wires=wires[i])


def ghz_rot(settings, wires):
    qml.Hadamard(wires=wires[0])
    for i in range(1, len(wires)):
        qml.CNOT(wires=[wires[0], wires[i]])

    for i in range(len(wires)):
        qml.Rot(*settings[3 * i : 3 * i + 3], wires=wires[i])


def max_entangled(settings, wires):
    qml.Hadamard(wires=[wires[0]])
    qml.CNOT(wires=wires[0:2])
    qml.Rot(*settings[0:3], wires=wires[0])


def ghz_prep_node(n):
    return [QNopt.PrepareNode(1, range(n), QNopt.ghz_state, 0)]


def arb_prep_node(n):
    return [QNopt.PrepareNode(1, range(n), qml.ArbitraryStatePreparation, 2 ** (n + 1) - 2)]


def local_rot_meas_nodes(n):
    return [QNopt.MeasureNode(2, 2, [i], local_rot, 3) for i in range(n)]


# Chain Network Ansatz Helpers
def chain_nlocal_max_entangled_prep_nodes(n):
    return [
        QNopt.PrepareNode(1, [2 * i, 2 * i + 1], QNopt.max_entangled_state, 3) for i in range(n)
    ]


def chain_nlocal_arbitrary_prep_nodes(n):
    return [
        QNopt.PrepareNode(1, [2 * i, 2 * i + 1], qml.ArbitraryStatePreparation, 6) for i in range(n)
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
        [
            QNopt.MeasureNode(
                2, 2, [2 * i + 1, 2 * i + 2], qml.adjoint(QNopt.max_entangled_state), 3
            )
            for i in range(n - 1)
        ]
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
        [
            QNopt.MeasureNode(
                2, 2, [2 * i + 1, 2 * i + 2], qml.templates.subroutines.ArbitraryUnitary, 15
            )
            for i in range(n - 1)
        ]
    )

    meas_nodes.append(
        QNopt.MeasureNode(
            2, 2, [2 * n - 1], lambda settings, wires: qml.Rot(*settings, wires=wires), 3
        )
    )
    return meas_nodes


# Star Network Ansatz Helper
def star_nlocal_max_entangled_prep_nodes(n):
    return [QNopt.PrepareNode(1, [i, n + i], max_entangled, 3) for i in range(n)]


def star_22_local_ry_meas_nodes(n):
    meas_nodes = [QNopt.MeasureNode(2, 2, [i], QNopt.local_RY, 1) for i in range(n)]
    meas_nodes.append(QNopt.MeasureNode(2, 2, [i for i in range(n, 2 * n)], QNopt.local_RY, n))

    return meas_nodes


def star_22_local_rot_meas_nodes(n):
    meas_nodes = [QNopt.MeasureNode(2, 2, [i], local_rot, 3) for i in range(n)]
    meas_nodes.append(QNopt.MeasureNode(2, 2, [i for i in range(n, 2 * n)], local_rot, 3 * n))

    return meas_nodes


def star_22_ghz_rot_meas_nodes(n):
    meas_nodes = [QNopt.MeasureNode(2, 2, [i], local_rot, 3) for i in range(n)]
    meas_nodes.append(
        QNopt.MeasureNode(2, 2, [i for i in range(n, 2 * n)], qml.adjoint(ghz_rot), 3 * n)
    )

    return meas_nodes
