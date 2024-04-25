"""service URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path

from service import views

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('hello/', views.hello, name="hello"),
    path('perf/doperf', views.perf_model, name="perf_model"),
    path('perf/list/all', views.list_perf_result, name="list_perf_result"),
    path('perf/predict/', views.predict_perf_result, name="do_predict"),
    path(r'perf/list/detail/', views.list_detail, name="list_detail"),
    path('perf/schedule/', views.get_schedule_info, name="list_schedule"),
]
