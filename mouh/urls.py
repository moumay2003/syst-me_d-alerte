from django.urls import path
from  . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [ 
     
      path('mm', views.forecast_view, name='forecast'),
      path('/i', views.plot_operations, name='plot_operations'),
      path('dd', views.index, name='index'),
      path('r', views.resultats_comparaison, name='resultats_comparaison'),
      path('', views.forecast_view2, name='forecast2'),
     
     

    ]