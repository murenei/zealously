from django import forms


class SimilarityForm(forms.Form):

    doc1 = forms.CharField(widget=forms.Textarea(
        attrs={
            'class': 'form-control',
            'id': 'formControlTextareaSimilarity1',
            'rows': '3',
            'required': True
        })
    )
