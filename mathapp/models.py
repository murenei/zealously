from django.db import models
from django.utils import timezone
import datetime

# Create your models here.


class Question(models.Model):

    QUESTION_TYPES = (
        ('QUAD', 'Quadratic'),
        ('LIN', 'Linear'),
        ('POLY', 'Rational Polynomial'),
    )
    question_text = models.CharField(max_length=200)
    solution_text = models.CharField(max_length=200)
    question_type = models.CharField(
        max_length=4,
        choices=QUESTION_TYPES,
        default='QUAD')
    created_date = models.DateTimeField('date created', auto_now_add=True)  # Supports a datetime.datetime instance

    def __str__(self):
        return "{}: {}".format(self.question_type, self.question_text)

    def was_published_recently(self):
        return self.pub_date >= timezone.now() - datetime.timedelta(days=1)

    class Meta:
        ordering = ('created_date',)
