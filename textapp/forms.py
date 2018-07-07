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


class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50, widget=forms.TextInput(
        attrs={'class': 'form-control'})
    )
    file = forms.FileField(widget=forms.FileInput(attrs={'class': 'form-control'}))
