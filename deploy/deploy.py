from flask import Flask, render_template, Response, request
from flask_bootstrap import Bootstrap
from predict import Predict

app = Flask(__name__, template_folder='./static/templates')
Bootstrap(app)

predict = Predict()
# nav=Nav()

# nav.register_element('top',Navbar(u'视频监控系统',
#                                     View(u'主页','index'),
#                                     Subgroup(u'监控',
#                                              View(u'项目一','index'),
#                                              Separator(),
#                                              View(u'项目二', 'index'),
#                                     ),
# ))

# nav.init_app(app)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
 
 
@app.route('/summary',methods=['post'])
def summary():
    #登录需要两个参数，name和pwd
    text = request.form.get('text')
    print(text)
    result = predict.predcit(text)
    print(result)

    return render_template('index.html', result=result)
    


# @app.route('/matplot_feed')
# def matplot_feed():
#     return Response(gen_plot(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)