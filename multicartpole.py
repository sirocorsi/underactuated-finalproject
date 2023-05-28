import numpy as np

from pydrake.all import DiagramBuilder, Simulator, MultibodyPlant, Parser, LinearQuadraticRegulator, Diagram, SceneGraph, PlanarSceneGraphVisualizer, LogVectorOutput, Context, DirectCollocation, Solve, PiecewisePolynomial, FiniteHorizonLinearQuadraticRegulatorOptions, MakeFiniteHorizonLinearQuadraticRegulator
import pydrake.symbolic as sym

def get_urdf(n_pend, l_pend = None, l_max = 1, name = None, color = "0 0 0.8", alpha = 1):
    """
    Returns a urdf for a multi-cartpole system with n_pend pendulums.
    The length of the pendulums is l_max/(i+1) for i in range(n_pend),
    or the values in l_pend if provided.
    """
    if name is None:
        name = f"{n_pend}-cartpole"

    cart_length = 0.2*(n_pend+1)

    if l_pend is None:
        l_pend = [l_max/(i+1) for i in range(n_pend)]

    cart_urdf = f"""
        <link name="cart">
        
        <inertial>
            <origin xyz="0 0 0" />
            <mass value="1" />
        </inertial>

        <visual>
            <origin xyz="0 0 0" />
            <geometry>
            <box size="{cart_length} .2 .2" />
            </geometry>
            <material>
            <color rgba="{color} {alpha}" />
            </material>
        </visual>

        <visual>
            <origin xyz="{cart_length/5} 0 -.1" rpy="{np.pi/2} 0 0" />
            <geometry>
            <cylinder radius=".05" length=".18"/>
            </geometry>
            <material>
            <color rgba="0.2 0.2 0.2 {alpha}" />
            </material>
        </visual>
        
        <visual>
            <origin xyz="{-cart_length/5} 0 -.1" rpy="{np.pi/2} 0 0" />
            <geometry>
            <cylinder radius=".05" length=".18"/>
            </geometry>
            <material>
            <color rgba="0.2 0.2 0.2 {alpha}" />
            </material>
        </visual>
        </link>

        <joint name="x" type="prismatic">
        <parent link="world" />
        <child link="cart" />
        <axis xyz="1 0 0" />
        </joint>

        <transmission type="SimpleTransmission" name="cart_force">
        <actuator name="force" />
        <joint name="x" />
        </transmission>
    """

    pendulum_urdf = ""
    for i in range(n_pend):
        pendulum_urdf += f""" 
        <link name="pendulum{i}">

            <inertial>
                <origin xyz="0 0 {l_pend[i]}" />
                <mass value="1" />
            </inertial>

            <visual>
            <origin xyz="0 0 {l_pend[i]}" />
            <geometry>
                <sphere radius=".05" />
            </geometry>
            <material>
                <color rgba="{color} {alpha}" />
            </material>
            </visual>

            <visual>
            <origin xyz="0 0 {l_pend[i]/2}" />
            <geometry>
                <cylinder radius="0.01" length="{l_pend[i]}" />
            </geometry>
            <material>
                <color rgba=".9 .9 .9 {alpha}" />
            </material>
            </visual>
        
        </link>

        <joint name="theta{i}" type="continuous">
            <origin xyz="{-cart_length/2 + cart_length/(n_pend+1)*(i+1)} 0 0.1" />
            <parent link="cart" />
            <child link="pendulum{i}" />
            <axis xyz="0 -1 0" />
        </joint>
        """

    urdf = f"""
    <?xml version="1.0"?>
        <robot name="{name}">
    {cart_urdf}
    {pendulum_urdf}
        </robot>
    </xml>
    """
    return urdf

def add_traj_stabilizer_finite_horizon_LQR(plant, builder, x_traj, u_traj, Q = None, R = None, name = "Finite Horizon LQR"):
    if Q is None:
       Q = np.diag([10.0]*plant.num_positions() + [1.0]*plant.num_velocities())
    if R is None: 
       R = np.eye(plant.num_actuated_dofs())
   
    options = FiniteHorizonLinearQuadraticRegulatorOptions()
    options.x0 = x_traj
    options.u0 = u_traj
    options.input_port_index = plant.get_actuation_input_port().get_index()
    options.Qf = Q

    context = plant.CreateDefaultContext()
    regulator = builder.AddSystem(MakeFiniteHorizonLinearQuadraticRegulator(plant, context, t0=options.u0.start_time(), tf=options.u0.end_time(), Q=Q, R=R, options=options))
    regulator.set_name(name)
    builder.Connect(regulator.get_output_port(0), plant.get_actuation_input_port())
    builder.Connect(plant.get_state_output_port(), regulator.get_input_port(0))
   
    return regulator

def swing_up(plant, context = None, N = 21, R = 0.1, x_lim = 1):
    if context is None:
       context = plant.CreateDefaultContext()

    min_dt = 0.05
    max_dt = 0.5

    dircol = DirectCollocation(system=plant, context=context, num_time_samples=N, minimum_timestep=min_dt, maximum_timestep=max_dt, input_port_index=plant.get_actuation_input_port().get_index())
    prog = dircol.prog()

    dircol.AddEqualTimeIntervalsConstraints()

    xf = np.zeros((plant.num_positions() + plant.num_velocities(),1))
    x0 = np.array([[0] + [np.pi]*(plant.num_positions()-1) + [0]*plant.num_velocities()]).T

    prog.AddBoundingBoxConstraint(x0, x0, dircol.initial_state())
    prog.AddBoundingBoxConstraint(xf, xf, dircol.final_state())

    dircol.AddConstraintToAllKnotPoints(-x_lim <= dircol.state()[0])
    dircol.AddConstraintToAllKnotPoints(dircol.state()[0] <= x_lim)

    dircol.AddRunningCost(R * dircol.input()[0] ** 2)
    dircol.AddFinalCost(dircol.time())

    initial_x_trajectory = PiecewisePolynomial.FirstOrderHold([0.0, 4.0], [x0, xf])
    dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

    result = Solve(prog)

    x_traj = dircol.ReconstructStateTrajectory(result)
    u_traj = dircol.ReconstructInputTrajectory(result)

    return result, x_traj, u_traj

def get_lqr_diagram(urdf, n_pend, **kwargs):
    """
    Returns a diagram with a multibody plant and a lqr controller.
    """

    builder = DiagramBuilder()
    scene_graph = SceneGraph()
    scene_graph.set_name("scene graph")

    plant = MultibodyPlant(time_step=0.0)
    plant.set_name("plant")

    builder.AddSystem(scene_graph)
    plant.RegisterAsSourceForSceneGraph(scene_graph)

    Parser(plant).AddModelsFromString(urdf, "urdf")
    plant.Finalize()
    builder.AddSystem(plant)
    builder.Connect(plant.get_output_port(0), scene_graph.get_source_pose_port(plant.get_source_id()))

    x_star = np.zeros(n_pend*2+2)
    Q = np.eye(n_pend*2+2)
    R = np.eye(1)
    plant_context = plant.CreateDefaultContext()
    plant.SetPositionsAndVelocities(plant_context, x_star)
    plant.get_actuation_input_port().FixValue(plant_context, [0])
    lqr = builder.AddSystem(LinearQuadraticRegulator(plant, plant_context, Q, R, input_port_index=plant.get_actuation_input_port().get_index()))

    builder.Connect(plant.get_state_output_port(), lqr.get_input_port(0))
    builder.Connect(lqr.get_output_port(0), plant.get_actuation_input_port())

    visualizer = builder.AddSystem(PlanarSceneGraphVisualizer(scene_graph, show=False, **kwargs))
    visualizer.set_name("visualizer")

    logger = LogVectorOutput(plant.get_state_output_port(), builder)
    logger.set_name("logger")
    builder.Connect(scene_graph.get_query_output_port(), visualizer.get_input_port(0))


    diagram = builder.Build()
    return diagram

def simulate_diagram(diagram, x0, t_max):
    """
    Simulates a diagram with a given initial state and time horizon.
    """

    simulator = Simulator(diagram)

    context = simulator.get_mutable_context()
    context.SetTime(0.)
    context.SetContinuousState(x0)

    simulator.set_publish_every_time_step(False)
    simulator.get_mutable_integrator().set_fixed_step_mode(True)

    simulator.Initialize()
    simulator.AdvanceTo(t_max)
    return context

def replace(expr, old_subexpr, new_subexpr):
    if not isinstance(expr, sym.Expression):
        return expr
    if expr.EqualTo(old_subexpr):
        return new_subexpr
    ctor, old_args = expr.Unapply()
    new_args = [
        replace(arg, old_subexpr, new_subexpr)
        for arg in old_args
    ]
    return ctor(*new_args)

def replace_sin(expr, q, s): # replaces sin(q) with s in expr
    sinq = sym.Expression.sin(q)
    return replace(expr, sinq, s)

def replace_cos(expr, q, c): # replaces cos(q) with c in expr
    cosq = sym.Expression.cos(q)
    return replace(expr, cosq, c)

def ManipulatorDynamics(plant, q, qd, context=None):
    if context is None:
        context = plant.CreateDefaultContext()
    plant.SetPositions(context, q)
    plant.SetVelocities(context, qd)
    
    M = plant.CalcMassMatrixViaInverseDynamics(context)
    C = plant.CalcBiasTerm(context)
    tauG = plant.CalcGravityGeneralizedForces(context)
    B = plant.MakeActuationMatrix()

    return M, C, tauG, B

def ManipulatorDynamicsToPolynomial(M, C, tauG, B, q, s, c):
    for q,s,c in zip(q,s,c):
        M = np.vectorize(replace_sin)(M, q, s)
        M = np.vectorize(replace_cos)(M, q, c)
        C = np.vectorize(replace_sin)(C, q, s)
        C = np.vectorize(replace_cos)(C, q, c)
        tauG = np.vectorize(replace_sin)(tauG, q, s)
        tauG = np.vectorize(replace_cos)(tauG, q, c)
        B = np.vectorize(replace_sin)(B, q, s)
        B = np.vectorize(replace_cos)(B, q, c)
    return M, C, tauG, B
