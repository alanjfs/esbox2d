#include <chrono>
#include <vector>

#include <Box2D/Box2D.h>

#include <Corrade/Utility/Arguments.h>
#include <Corrade/Utility/DebugStl.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/ConfigurationValue.h>
#include <Magnum/Math/DualComplex.h>
#include <Magnum/Math/Angle.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/Primitives/Square.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/TranslationRotationScalingTransformation2D.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/Shaders/Flat.h>
#include <Magnum/Trade/MeshData2D.h>
#include <Magnum/Timeline.h>
#include <Magnum/Math/Constants.h> // pi

#include <Eigen/Core>   // Eigen::VectorXd etc.
#include <random>       // std::normal_distribution

namespace Magnum { namespace Examples {

typedef SceneGraph::Object<SceneGraph::TranslationRotationScalingTransformation2D> Object2D;
typedef SceneGraph::Scene<SceneGraph::TranslationRotationScalingTransformation2D> Scene2D;
typedef Math::Constants<double> C;

using namespace Math::Literals;

#include "util.inl"
#include "es.inl"


struct StepInfo {
    Eigen::Matrix<double, 4, 1> observation;
    double reward;
    bool done;
};


class BoxES: public Platform::Application {
    public:
        explicit BoxES(const Arguments& arguments);

    private:
        void drawEvent() override;
        void mousePressEvent(MouseEvent& event) override;
        void mouseMoveEvent(MouseMoveEvent& event) override;
        void mouseReleaseEvent(MouseEvent& event) override;
        void keyPressEvent(KeyEvent& event) override;

        StepInfo step(unsigned int action);
        Eigen::Matrix<double, 4, 1> reset();
        void render();
        void close();

        void run();
        double run_episode(Eigen::Matrix<double, 4, 2> weights);

        b2Body* create_body(Object2D& object,
                            const Vector2& size,
                            b2BodyType type,
                            const DualComplex& tm,
                            Float density = 1.0f);

        GL::Mesh m_mesh{NoCreate};
        Shaders::Flat2D m_shader{NoCreate};

        Scene2D m_scene;
        Object2D* m_cameraobj;
        SceneGraph::Camera2D* m_camera;
        SceneGraph::DrawableGroup2D m_drawables;
        Containers::Optional<b2World> m_world;

        // Bodies
        b2Body* m_ground_body;
        b2Body* m_cart_body;
        b2Body* m_pole_body;

        // Eigen::VectorXd m_state;
        Eigen::Matrix<double, 4, 1> m_state;
        Eigen::Matrix<double, 4, 1> m_observation_space;
        Eigen::Matrix<int, 2, 1> m_action_space;
        unsigned int m_pop { 10 };

        int steps_beyond_done { -1 };
        bool m_disable_on_done { false };

        unsigned int _draw_count { 0 };

        double cart_position_threshold { 2.4 };
        double pole_angle_threshold_radians {
            12 * 2 * C::pi() / 360
        };

        unsigned int m_input_size { 4 };
        unsigned int m_output_size { 2 };

        Eigen::Matrix<double, 4, 2> m_W;

        // Joints
        b2Vec2 m_mouseworld { 0.0f, 0.0f };
        b2MouseJoint* m_mousejoint { nullptr };

        b2PrismaticJoint* m_cartjoint { nullptr };
        b2RevoluteJoint* m_polejoint { nullptr };

        Vector2 _dpiScaling;
};


BoxES::BoxES(const Arguments& arguments): Platform::Application{arguments, NoCreate} {
    {
        const Vector2 dpiScaling = this->dpiScaling({});
        Configuration conf;
        conf.setTitle("Magnum Box2D Example")
            .setSize(conf.size(), dpiScaling);
        GLConfiguration glConf;
        glConf.setSampleCount(dpiScaling.max() < 2.0f ? 8 : 2);
        if(!tryCreate(conf, glConf))
            create(conf, glConf.setSampleCount(0));
    }

    glfwGetWindowContentScale(this->window(), &_dpiScaling.x(), &_dpiScaling.y());

    /* Create the shader and the box mesh */
    m_shader = Shaders::Flat2D{};
    m_mesh = MeshTools::compile(Primitives::squareSolid());

    this->reset();

    setSwapInterval(1);

    Debug() << "Mouse to interact";
    Debug() << "Enter to run";
    Debug() << "R to reset";
    Debug() << "D to disable on done";
}


b2Body* BoxES::create_body(Object2D& object,
                           const Vector2& halfSize,
                           const b2BodyType type,
                           const DualComplex& tm,
                           const Float density) {
    b2BodyDef bodyDefinition;
    bodyDefinition.position.Set(tm.translation().x(), tm.translation().y());
    bodyDefinition.angle = Float(tm.rotation().angle());
    bodyDefinition.type = type;
    bodyDefinition.linearDamping = 0.0f;
    bodyDefinition.angularDamping = 1.0f;
    b2Body* body = m_world->CreateBody(&bodyDefinition);
    b2PolygonShape shape;
    shape.SetAsBox(halfSize.x(), halfSize.y());

    b2FixtureDef fixture;
    fixture.friction = 0.8f;
    fixture.density = density;
    fixture.shape = &shape;
    body->CreateFixture(&fixture);

    body->SetUserData(&object);
    object.setScaling(halfSize);

    return body;
}


Eigen::Matrix<double, 4, 1> BoxES::reset() {
    // Erase everything first
    for (Object2D* obj = m_scene.children().first(); obj;) {
        Object2D* next = obj->nextSibling();
        delete obj;
        obj = next;
    }

    /* Configure camera */
    m_cameraobj = new Object2D{ &m_scene };
    m_camera = new SceneGraph::Camera2D{ *m_cameraobj };
    m_camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
        .setProjectionMatrix(Matrix3::projection({10.0f, 10.0f}))
        .setViewport(GL::defaultFramebuffer.viewport().size());

    m_world.emplace(b2Vec2{0.0f, -98.1f});

    /* Create ground */
    auto ground = new Object2D{&m_scene};
    m_ground_body = create_body(
        *ground,
        { 10.0f, 0.1f },
        b2_staticBody,
        DualComplex::translation(Vector2::yAxis(-2.0f))
    );

    new BoxDrawable{*ground, m_mesh, m_shader, 0xa5c9ea_rgbf, m_drawables};
    const DualComplex gtm = DualComplex::translation({ 0.0f, 1.0f });

    auto cart = new Object2D{ &m_scene };
    auto pole = new Object2D{ &m_scene };

    // Cart
    {
        const Vector2 size { 0.5f, 0.3f };
        const float density { 1.0f };
        const DualComplex tm = gtm * DualComplex::translation({ 0.0f, 0.0f });
        m_cart_body = create_body(*cart, size, b2_dynamicBody, tm, density);
        new BoxDrawable(*cart, m_mesh, m_shader, 0x80d8ee_rgbf, m_drawables);
    }

    // Pole
    {
        const Vector2 size { 0.075f, 0.5f };
        const float density { 0.1f };
        const DualComplex tm = (
            gtm *
            DualComplex::rotation({ 1.0_degf }) *
            DualComplex::translation({ 0.0f, size.y() })
        );
        m_pole_body = create_body(*pole, size, b2_dynamicBody, tm, density);
        new BoxDrawable(*pole, m_mesh, m_shader, 0x2f83cc_rgbf, m_drawables);
    }

    b2PrismaticJointDef cart_joint;
    cart_joint.Initialize(
        m_ground_body,
        m_cart_body,
        { 0.0f, 0.5f },
        b2Vec2(1.0f, 0.0f)
    );

    cart_joint.lowerTranslation = -4.0f;
    cart_joint.upperTranslation = 4.0f;
    cart_joint.enableLimit = true;

    b2RevoluteJointDef pole_joint;
    pole_joint.Initialize(m_cart_body, m_pole_body, m_cart_body->GetWorldCenter());

    m_cartjoint = (b2PrismaticJoint*)m_world->CreateJoint(&cart_joint);
    m_polejoint = (b2RevoluteJoint*)m_world->CreateJoint(&pole_joint);

    std::default_random_engine g;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    this->m_state = {
        uniform(g),
        uniform(g),
        uniform(g),
        uniform(g)
    };

    this->steps_beyond_done = -1;

    return m_state;
}


class QueryCallback : public b2QueryCallback {
public:
    QueryCallback(const b2Vec2& point) {
        m_point = point;
        m_fixture = nullptr;
    }

    bool ReportFixture(b2Fixture* fixture) override {
        b2Body* body = fixture->GetBody();
        if (body->GetType() == b2_dynamicBody)
        {
            bool inside = fixture->TestPoint(m_point);
            if (inside)
            {
                m_fixture = fixture;

                // We are done, terminate the query.
                return false;
            }
        }

        // Continue the query.
        return true;
    }

    b2Vec2 m_point;
    b2Fixture* m_fixture;
};


void BoxES::mousePressEvent(MouseEvent& event) {
    if (event.button() == MouseEvent::Button::Left) {
        const auto pos = (
            m_camera->projectionSize() *
            Vector2::yScale(-1.0f) * (
                Vector2{ event.position() } / 
                Vector2{ windowSize() } - Vector2{ 0.5f }
            )
        );

        b2Vec2 p { pos.x(), pos.y() };
        m_mouseworld = p;

        if (m_mousejoint != nullptr) {
            return;
        }

        // Make a small box.
        // From https://github.com/erincatto/box2d/blob/f5437a1415a363a89c286657839b48a03eb61d20/testbed/test.cpp#L151
        b2AABB aabb;
        b2Vec2 d;
        d.Set(0.001f, 0.001f);
        aabb.lowerBound = p - d;
        aabb.upperBound = p + d;

        // Query the world for overlapping shapes.
        QueryCallback callback(p);
        m_world->QueryAABB(&callback, aabb);

        if (callback.m_fixture) {
            b2Body* body = callback.m_fixture->GetBody();
            b2MouseJointDef md;
            md.bodyA = m_ground_body;
            md.bodyB = body;
            md.target = p;
            md.maxForce = 1000.0f * body->GetMass();
            m_mousejoint = (b2MouseJoint*)m_world->CreateJoint(&md);
            body->SetAwake(true);
        }
    }

    else if (event.button() == MouseEvent::Button::Right) {

        /* Calculate mouse position in the Box2D world. Make it relative to window,
           with origin at center and then scale to world size with Y inverted. */
        const auto position = (
            m_camera->projectionSize() *
            Vector2::yScale(-1.0f) *
            (
                Vector2{ event.position() } /
                Vector2{ windowSize() } -
                Vector2{ 0.5f }
            )
        );

        auto destroyer = new Object2D{ &m_scene };
        create_body(*destroyer, {0.5f, 0.5f}, b2_dynamicBody, DualComplex::translation(position), 2.0f);
        new BoxDrawable{ *destroyer, m_mesh, m_shader, 0xffff66_rgbf, m_drawables };
    }
}

void BoxES::mouseMoveEvent(MouseMoveEvent& event) {
    const auto pos = (
        m_camera->projectionSize() *
        Vector2::yScale(-1.0f) * (
            Vector2{ event.position() } / 
            Vector2{ windowSize() } - Vector2{ 0.5f }
        )
    );

    b2Vec2 p { pos.x(), pos.y() };

    m_mouseworld = p;
    
    if (m_mousejoint != nullptr) {
        m_mousejoint->SetTarget(p);
    }
}

void BoxES::mouseReleaseEvent(MouseEvent& event) {
    if (m_mousejoint != nullptr) {
        m_world->DestroyJoint(m_mousejoint);
        m_mousejoint = nullptr;
    }
}


void BoxES::keyPressEvent(KeyEvent& event) {
    if (event.key() == KeyEvent::Key::R) {
        this->reset();
    }

    if (event.key() == KeyEvent::Key::D) {
        this->m_disable_on_done ^= true;
    }

    if (event.key() == KeyEvent::Key::Enter) {
        auto begin = std::chrono::steady_clock::now(); {
            this->run();
        }
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        Debug() << "run() took" << duration * 0.001f << "milliseconds";
        
    }

    // Run a performance test
    if (event.key() == KeyEvent::Key::Space) {
        this->reset();

        auto begin = std::chrono::steady_clock::now();
        for (int i=0; i < 200; i++) {
            this->step(0);
        }
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        Debug() << "200 steps took" << duration * 0.001f << "milli seconds";
    }

    if (event.key() == KeyEvent::Key::Backspace) {
        Eigen::Matrix<double, 4, 2> w = randn(4, 2);
        this->run_episode(w);
    }

}


// Implementation of numpy.argmax([])
unsigned int extract_move(Eigen::VectorXd action) {
    double largest = 0;
    int j = 0;

    for (int i=0; i < action.size(); i++) {
        if (action[i] > j) largest = i;
    }

    return largest;
}



double BoxES::run_episode(Eigen::Matrix<double, 4, 2> weights) {
    Eigen::Matrix<double, 4, 1> obs = this->reset();
    double episode_reward { 0.0 };
    unsigned int step { 0 };
    bool done { false };
    unsigned int max_steps { 200 };

    while (!done) {
        unsigned int move = 0;
        if (step < max_steps) {
            Eigen::Matrix<double, 2, 1> action = weights.transpose() * obs;
            move = extract_move(action);
        }
        else {
            move = 0; // Do nothing
        }

        StepInfo info = this->step(move);

        obs = info.observation;
        episode_reward += info.reward;
        done = info.done;
        step += 1;
    }

    return episode_reward;
}


#define NPOP 50
#define ISIZE 4
#define OSIZE 2

void BoxES::run() {
    for (int gen=0; gen < 100; gen++) {
        double gen_eval = run_episode(m_W);

        const float sigma { 0.1f };
        const float alpha { 0.001f };

        std::array<Eigen::Matrix<double, ISIZE, OSIZE>, NPOP> N;
        Eigen::VectorXd R(NPOP);

        for (auto& n : N) { n = randn(ISIZE, OSIZE); }

        for (int j=0; j < NPOP; j++) {
            Eigen::Matrix<double, ISIZE, OSIZE> w_try = m_W + N[j] * sigma;
            R(j) = run_episode(w_try);
        }

        // This should be (N.transpose() * R).transpose()
        // But N is an array rather than a tensor..
        // https://gist.github.com/Adriel-M/4daabe115982fe1d9159e730ac3f79a5#file-cartpole-els-py-L121
        Eigen::Matrix<double, 4, 2> weighted_weights;
        for (int i=0; i < N.size(); i++) {
            N[i] *= R[i];
        }

        m_W += alpha / (NPOP * sigma) * weighted_weights;
        double gen_mean = R.mean();

        printf("Generation: %d, Return: %.2f, Population Mean: %.2f\n",
               gen, gen_eval, gen_mean);
    }
}


StepInfo BoxES::step(unsigned int action) {
    /* Step the world and step all object positions */
    static int substeps = 1;
    static float time_step = 1.0f / 60.0f / substeps;
    static int velocity_iterations = 6;
    static int position_iterations = 2;

    b2Vec2 force { action == 1 ? 10.0f : -10.0f, 0 };
    m_cart_body->ApplyForceToCenter(force, true); 

    auto begin = std::chrono::steady_clock::now();

    for (int i=0; i < substeps; i++) {
        m_world->Step(
            time_step,
            velocity_iterations,
            position_iterations
        );
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    for (b2Body* body = m_world->GetBodyList(); body; body = body->GetNext()) {
        (*static_cast<Object2D*>(body->GetUserData()))
            .setTranslation({body->GetPosition().x, body->GetPosition().y})
            .setRotation(Complex::rotation(Rad(body->GetAngle())));
    }

    double cart_position = m_cart_body->GetPosition().x;
    double cart_velocity = m_cart_body->GetLinearVelocity().x;
    double pole_angle = m_pole_body->GetAngle();
    double pole_angular_velocity = m_pole_body->GetAngularVelocity();

    bool done = (
        cart_position < -this->cart_position_threshold ||
        cart_position > this->cart_position_threshold ||
        pole_angle < -this->pole_angle_threshold_radians ||
        pole_angle > this->pole_angle_threshold_radians
    );

    double reward { 0.0 };

    if (!done)
        reward = 1.0;
    else if (this->steps_beyond_done == -1) {
        // Pole just fell!
        this->steps_beyond_done = 0;
        reward = 1.0;
    } else {
        if (this->steps_beyond_done == 0) {
            Debug() << "Don't call step() after done";
        }
        this->steps_beyond_done += 1;
        reward = 0.0;

        if (this->m_disable_on_done) {
            m_cart_body->SetEnabled(false);
            m_pole_body->SetEnabled(false);
        }
    }

    StepInfo info;
    info.observation = {
        cart_position,
        cart_velocity,
        pole_angle,
        pole_angular_velocity
    };
    info.reward = reward;
    info.done = done;

    return info;
}


void BoxES::render() {
    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color);
    m_camera->draw(m_drawables);
    swapBuffers();
}


void BoxES::drawEvent() {
    this->step(0);
    this->render();
    redraw();
    _draw_count++;
}

}}

MAGNUM_APPLICATION_MAIN(Magnum::Examples::BoxES)
