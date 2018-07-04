from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect

# Create your views here.


from mathgen import algebra


def index(request):
    template = "index.html"
    return render(request, template)


def gen_equation(request):
    problem_type = request.POST['problem_type']
    n_problems = request.POST['n_problems']

    eqs = []
    for i in range(int(n_problems)):
        if problem_type == 'QUAD':
            eq, s = algebra.make_quadratic_eq()
        elif problem_type == 'LIN':
            eq, s = algebra.make_linear_eq()
        else:
            eq, s = algebra.make_rational_poly_simplify()
        print('Equation ' + str(i) + ': ' + eq)
        eqs.append(eq)

    context = {
        'eqs': eqs
    }
    # print(len(eqs))

    template = "index.html"

    return render(request, template, context)


# ================================================
# CLASS BASED (+ generic) VIEWS
from django.views.generic import TemplateView


class AboutView(TemplateView):
    template_name = "about.html"


# ================================================ #
# REST FRAMEWORK


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import authentication, permissions
from django.contrib.auth.models import User


class ListUsers(APIView):
    """
    View to list all users in the system.

    * Requires token authentication.
    * Only admin users are able to access this view.
    """
    authentication_classes = (authentication.TokenAuthentication,)
    permission_classes = (permissions.IsAdminUser,)

    def get(self, request, format=None):
        """
        Return a list of all users.
        """
        usernames = [user.username for user in User.objects.all()]
        return Response(usernames)


from .models import Question


class ListQuestions(APIView):
    """
    View to list all questions in the system.

    * Does not require token authentication.
    * Anyone can access this view.
    """
    authentication_classes = []
    permission_classes = []

    def get(self, request, format=None):
        """
        Return a list of all Questions.
        """
        questions = [q.question_text for q in Question.objects.all()]
        return Response(questions)
