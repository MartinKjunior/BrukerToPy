
# Sections Q: Quantities

## <a id="MR signal quantities"></a> MR signal quantities
The items in this group are related to the MR signal and quantities of the MR sequence used to acquire the signal.

| Code | OSIPI name| Alternative names|Notation|Description|OSIPI units|Reference|
| -- | -- | -- | -- | -- | -- | -- |
| Q.MS1.001.[j] <button class="md-button md-button--hyperlink">HYPERLINK</button> | <a id="S"></a> Signal | -- | *S<sub>j</sub>* | The MR signal (magnitude, phase or complex depending on context) in compartment *j*. | a.u. | -- |
| Q.MS1.002.[j] | <a id="S_BL"></a> Baseline signal | -- | *S<sub>BL,j</sub>* | Pre-contrast MR signal (magnitude, phase or complex depending on context) in compartment *j* before the arrival of indicator at the tissue of interest. | a.u. | -- |
| Q.MS1.999 | <a id="not listed MS1"></a> Quantity not listed | -- | -- | This is a custom free-text item, which can be used if a quantity of interest is not listed. Please state a literature reference and request the item to be added to the lexicon for future usage. | [variable] | -- |

## <a id="Electromagnetic quantities"></a> Electromagnetic quantities
The items in this group are related to electromagnetic tissue properties and electromagnetic properties of contrast agents. The abbreviations SE and GE denote spin and gradient echo.

| Code | OSIPI name | Alternative names|Notation|Description|OSIPI units|Reference|
| -- | -- | -- | -- | -- | -- | -- |
| Q.EL1.001.[j] | <a id="R1"></a> Longitudinal relaxation rate | *R<sub>1</sub>* -relaxation rate | $R_{1,j}$ | Longitudinal relaxation rate in compartment *j*. | 1/s | -- |
| Q.EL1.002.[j] | <a id="R10"></a> Native longitudinal relaxation rate |Baseline *R<sub>1</sub>* | $R_{10,j}$ | Longitudinal relaxation rate in compartment *j*. | 1/s | -- |
| Q.EL1.003.[j] | <a id="DeltaR1"></a> Change in longitudinal relaxation rate  | -- | $\Delta R_{1,j}^*$ | Change in longitudinal relaxation rate with respect to $R_{10,j}$ in compartment *j*. | 1/s | -- |
| Q.EL1.999 | <a id="not listed EL1"></a> Quantity not listed | -- | -- | This is a custom free-text item, which can be used if a quantity of interest is not listed. Please state a literature reference and request the item to be added to the lexicon for future usage. | [variable] | -- |

## <a id="Indicator concentration quantities"></a> Indicator concentration quantities
The items of this group of quantities are either measured or modeled quantities used when pharmacokinetic modeling is applied. This section is split into the subsections indicator kinetic model quantities and AIF model quantities. The latter contains only quantities specific to often used AIF models.

### <a id="Indicator kinetic model quantities"></a>  Indicator kinetic model quantities
The items of this group of quantities are either measured or modeled quantities used when pharmacokinetic modeling is applied.

| Code | OSIPI name| Alternative names|Notation|Description|OSIPI units|Reference|
| -- | -- | -- | -- | -- | -- | -- |
| Q.IC1.001.[j] | <a id="C"></a> Indicator concentration | -- | *C<sub>j</sub>* | Concentration of indicator molecules in compartment *j*. | mM | -- |
| Q.IC1.999 | <a id="not listed IC1"></a> Quantity not listed | -- | -- | This is a custom free-text item, which can be used if a quantity of interest is not listed. Please state a literature reference and request the item to be added to the lexicon for future usage. | [variable] | -- |