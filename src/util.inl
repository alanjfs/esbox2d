
class BoxDrawable: public SceneGraph::Drawable2D {
    public:
        explicit BoxDrawable(Object2D& object, GL::Mesh& mesh, Shaders::Flat2D& shader, const Color4& color, SceneGraph::DrawableGroup2D& drawables): SceneGraph::Drawable2D{object, &drawables}, m_mesh(mesh), m_shader(shader), _color{color} {
        }

    private:
        void draw(const Matrix3& transformationMatrix, SceneGraph::Camera2D& camera) override {
            m_shader
                .setTransformationProjectionMatrix(camera.projectionMatrix()*transformationMatrix)
                .setColor(_color);
            m_mesh.draw(m_shader);
        }

        GL::Mesh& m_mesh;
        Shaders::Flat2D& m_shader;
        Color4 _color;
};
