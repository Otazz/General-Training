from aiohttp import web
from routes import setup_routes
import os
import aiohttp_jinja2
import jinja2

routes = web.RouteTableDef()

@routes.get('/')
async def hello(request):
    return aiohttp_jinja2.render_template('main.html',
                                              request,
                                              {})

PROJECT_ROOT = os.getcwd()



def setup_static_routes(app):
   app.router.add_static('/static/',
                         path=PROJECT_ROOT / 'static',
                         name='static')

app = web.Application()
print(type(app))
aiohttp_jinja2.setup(
    app, loader=jinja2.PackageLoader('static'))
app.add_routes(routes)
#setup_routes(app)
#setup_static_routes(app)
web.run_app(app, host='127.0.0.1', port=8080)