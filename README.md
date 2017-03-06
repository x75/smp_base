

# Sensorimotor learning basic lib

A python library of code I use for sensorimotor learning experiments
with simulations, robots, etc. Performing incremental clean-up and
pulling from existing private repository. The basis idea is to
generate states (sensors) with actions (motors), learn different
predictive models from that data to approximate the sensorimotor
dynamics and use the models to compute future actions.


## reservoir lib

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">reservoirs.py</td>
<td class="org-left">contains Reservoir class, LearningRules class, a  few utility functions and a main method</td>
</tr>


<tr>
<td class="org-left">&#xa0;</td>
<td class="org-left">that demonstrates basic use of the class. It can definitely be simplified (WiP)</td>
</tr>


<tr>
<td class="org-left">learners.py</td>
<td class="org-left">this model embeds the underlying adaptive model into the sensorimotor context</td>
</tr>


<tr>
<td class="org-left">eligibility.py</td>
<td class="org-left">basic eligibility windows used in a variant of learning rules</td>
</tr>


<tr>
<td class="org-left">smp\_thread.py</td>
<td class="org-left">thread wrapper that provides constant dt run loop and asynchronous sensor callbacks</td>
</tr>
</tbody>
</table>

This repository is in an early stages of release which I push on the
occasion of sharing the smp\_sphero code <sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup>.


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> <https://github.com/x75/smp_sphero>
