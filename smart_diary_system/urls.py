"""smart_diary_system URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
import smart_diary_system.apis as apis

app_name = 'smart_diary'
urlpatterns = [
    #url(r'^admin/', admin.site.urls),
    url(r'^api/user$', apis.manage_user),
    url(r'^api/user/(?P<option>\w+)$', apis.manage_user),
    url(r'^api/diary$', apis.manage_diary),
    url(r'^api/diary/(?P<option>\w+)$', apis.manage_diary),
    url(r'^api/c_text$', apis.manage_c_text),
    url(r'^api/c_text/(?P<option>\w+)$', apis.manage_c_text),
    # url(r'^api/sentence$', apis.manage_sentence),
    # url(r'^api/sentence/(?P<option>\w+)$', apis.manage_sentence),
    url(r'^api/semantic$', apis.analyze_semantic),
    url(r'^api/semantic/(?P<option>\w+)$', apis.analyze_semantic),
    url(r'^api/nlp_ko_dict$', apis.manage_nlp_ko_dict),
    # url(r'^api/upload$', apis.manage_file),
    url(r'^api/download$', apis.download),
]
