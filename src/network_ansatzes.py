import pennylane as qml

import qnetvo as qnet


def local_rot(settings, wires):
    for i in range(len(wires)):
        qml.Rot(*settings[3 * i : 3 * i + 3], wires=wires[i])


def ghz_rot(settings, wires):
    for i in range(len(wires)):
        qml.Rot(*settings[3 * i : 3 * i + 3], wires=wires[i])

    for i in range(1, len(wires)):
        qml.CNOT(wires=[wires[0], wires[i]])

    for i in range(len(wires)):
        qml.Rot(*settings[3 * i : 3 * i + 3], wires=wires[i])


def max_entangled(settings, wires):
    qml.Hadamard(wires=[wires[0]])
    qml.CNOT(wires=wires[0:2])
    qml.Rot(*settings[0:3], wires=wires[0])


def psi_plus_state(settings, wires):
    qnet.ghz_state(settings, wires=wires)
    qml.PauliX(wires=wires[0])


def ryrz_cnot(settings, wires):
    qml.RY(settings[0], wires=wires[0])
    qml.RZ(settings[1], wires=wires[0])
    qml.CNOT(wires=wires[0:2])


def local_ry_cnot(settings, wires):
    qml.RY(settings[0], wires=wires[0])
    qml.RY(settings[1], wires=wires[1])
    qml.CNOT(wires=wires[0:2])


def ghz_prep_node(n):
    return [qnet.PrepareNode(1, range(n), qnet.ghz_state, 0)]


def arb_prep_node(n):
    return [qnet.PrepareNode(1, range(n), qml.ArbitraryStatePreparation, 2 ** (n + 1) - 2)]


def local_rot_meas_nodes(n):
    return [qnet.MeasureNode(2, 2, [i], local_rot, 3) for i in range(n)]


# Chain Network Ansatz Helpers
def chain_nlocal_max_entangled_prep_nodes(n):
    return [qnet.PrepareNode(1, [2 * i, 2 * i + 1], qnet.max_entangled_state, 3) for i in range(n)]


def chain_ryrz_cnot_prep_nodes(n):
    return [qnet.PrepareNode(1, [2 * i, 2 * i + 1], ryrz_cnot, 2) for i in range(n)]


def chain_ghz_prep_nodes(n):
    return [qnet.PrepareNode(1, [2 * i, 2 * i + 1], qnet.ghz_state, 0) for i in range(n)]


def chain_psi_plus_prep_nodes(n):
    return [
        qnet.PrepareNode(1, [2 * i, 2 * i + 1], psi_plus_state, 0) for i in range(n)
    ]

def chain_nlocal_arbitrary_prep_nodes(n):
    return [
        qnet.PrepareNode(1, [2 * i, 2 * i + 1], qml.ArbitraryStatePreparation, 6) for i in range(n)
    ]


def chain_local_ry_meas_nodes(n):
    meas_nodes = []

    meas_nodes.append(qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1))
    meas_nodes.extend(
        [qnet.MeasureNode(2, 2, [2 * i + 1, 2 * i + 2], qnet.local_RY, 2) for i in range(n - 1)]
    )
    meas_nodes.append(qnet.MeasureNode(2, 2, [2 * n - 1], qnet.local_RY, 1))

    return meas_nodes


def chain_local_rot_meas_nodes(n):
    meas_nodes = []
    meas_nodes.append(
        qnet.MeasureNode(2, 2, [0], lambda settings, wires: qml.Rot(*settings, wires=wires), 3)
    )

    meas_nodes.extend(
        [qnet.MeasureNode(2, 2, [2 * i + 1, 2 * i + 2], local_rot, 6) for i in range(n - 1)]
    )

    meas_nodes.append(
        qnet.MeasureNode(
            2, 2, [2 * n - 1], lambda settings, wires: qml.Rot(*settings, wires=wires), 3
        )
    )
    return meas_nodes


def chain_bell_meas_nodes(n):
    meas_nodes = []
    meas_nodes.append(
        qnet.MeasureNode(2, 2, [0], lambda settings, wires: qml.Rot(*settings, wires=wires), 3)
    )

    meas_nodes.extend(
        [
            qnet.MeasureNode(2, 2, [2 * i + 1, 2 * i + 2], qml.adjoint(qnet.max_entangled_state), 3)
            for i in range(n - 1)
        ]
    )

    meas_nodes.append(
        qnet.MeasureNode(
            2, 2, [2 * n - 1], lambda settings, wires: qml.Rot(*settings, wires=wires), 3
        )
    )
    return meas_nodes


def chain_arb_meas_nodes(n):
    meas_nodes = []
    meas_nodes.append(
        qnet.MeasureNode(2, 2, [0], lambda settings, wires: qml.Rot(*settings, wires=wires), 3)
    )

    meas_nodes.extend(
        [
            qnet.MeasureNode(
                2, 2, [2 * i + 1, 2 * i + 2], qml.templates.subroutines.ArbitraryUnitary, 15
            )
            for i in range(n - 1)
        ]
    )

    meas_nodes.append(
        qnet.MeasureNode(
            2, 2, [2 * n - 1], lambda settings, wires: qml.Rot(*settings, wires=wires), 3
        )
    )
    return meas_nodes


# Star Network Ansatz Helper
def star_nlocal_max_entangled_prep_nodes(n):
    return [qnet.PrepareNode(1, [i, n + i], qnet.max_entangled_state, 3) for i in range(n)]


def star_nlocal_arb_prep_nodes(n):
    return [qnet.PrepareNode(1, [i, n + i], qml.ArbitraryStatePreparation, 6) for i in range(n)]


def star_ghz_prep_nodes(n):
    return [qnet.PrepareNode(1, [i, n + i], qnet.ghz_state, 0) for i in range(n)]


def star_ryrz_cnot_prep_nodes(n):
    return [qnet.PrepareNode(1, [i, n + i], ryrz_cnot, 2) for i in range(n)]


def star_22_local_ry_meas_nodes(n):
    meas_nodes = [qnet.MeasureNode(2, 2, [i], qnet.local_RY, 1) for i in range(n)]
    meas_nodes.append(qnet.MeasureNode(2, 2, [i for i in range(n, 2 * n)], qnet.local_RY, n))

    return meas_nodes


def star_22_local_rot_meas_nodes(n):
    meas_nodes = [qnet.MeasureNode(2, 2, [i], local_rot, 3) for i in range(n)]
    meas_nodes.append(qnet.MeasureNode(2, 2, [i for i in range(n, 2 * n)], local_rot, 3 * n))

    return meas_nodes


def star_22_ghz_rot_meas_nodes(n):
    meas_nodes = [qnet.MeasureNode(2, 2, [i], local_rot, 3) for i in range(n)]
    meas_nodes.append(
        qnet.MeasureNode(2, 2, [i for i in range(n, 2 * n)], qml.adjoint(ghz_rot), 3 * n)
    )

    return meas_nodes
