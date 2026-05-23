from __future__ import annotations

from desktop_app.session_analysis import SessionResult


def _mitbih_distribution_text(result: SessionResult) -> str:
    counts = getattr(result, "mitbih_counts", {}) or {}
    pcts = getattr(result, "mitbih_percentages", {}) or {}

    names = {
        "S": "Supraventricular ectopic",
        "V": "Ventricular ectopic",
        "F": "Fusion beat",
        "Q": "Unknown / low confidence",
    }

    parts: list[str] = []
    for label in ("S", "V", "F", "Q"):
        if counts.get(label, 0) > 0:
            parts.append(f"{names.get(label, label)}: {float(pcts.get(label, 0.0)):.1f}%")

    return " | ".join(parts)


def _rate_note_en(result: SessionResult) -> str:
    if result.bpm_avg is None:
        return "Heart-rate statistics were not available for this source."

    avg = float(result.bpm_avg)

    bpm_range = ""
    if result.bpm_min is not None and result.bpm_max is not None:
        bpm_range = f" The observed heart-rate range was approximately {result.bpm_min:.0f}-{result.bpm_max:.0f} BPM."

    if getattr(result, "sustained_tachy_detected", False):
        return (
            f"The session met the sustained tachycardia criterion, with heart rate remaining above "
            f"110 BPM for about {result.tachy_longest_sec:.0f} seconds.{bpm_range}"
        )

    if getattr(result, "recurrent_high_rate_detected", False):
        return (
            f"The heart rate was repeatedly above the expected range, totaling about "
            f"{getattr(result, 'high_rate_total_sec', 0.0):.0f} seconds above 100 BPM."
            f"{bpm_range}"
        )

    if getattr(result, "recurrent_low_rate_detected", False):
        return (
            f"The heart rate was repeatedly below the expected range, totaling about "
            f"{getattr(result, 'low_rate_total_sec', 0.0):.0f} seconds below 60 BPM."
            f"{bpm_range}"
        )

    if avg > 100:
        return f"The average heart rate was elevated at approximately {avg:.0f} BPM.{bpm_range}"

    if avg < 60:
        return f"The average heart rate was low at approximately {avg:.0f} BPM.{bpm_range}"

    return f"The average heart rate was approximately {avg:.0f} BPM and remained within an expected range for this recording.{bpm_range}"


def _rate_note_ar(result: SessionResult) -> str:
    if result.bpm_avg is None:
        return "إحصاءات معدل النبض غير متوفرة لهذا المصدر."

    avg = float(result.bpm_avg)

    bpm_range = ""
    if result.bpm_min is not None and result.bpm_max is not None:
        bpm_range = f" وكان مدى النبض المسجل تقريبًا بين {result.bpm_min:.0f} و {result.bpm_max:.0f} ض/د."

    if getattr(result, "sustained_tachy_detected", False):
        return (
            f"تحقّق معيار تسرّع القلب المستمر، حيث بقي معدل النبض أعلى من 110 ض/د لمدة "
            f"{result.tachy_longest_sec:.0f} ثانية تقريبًا.{bpm_range}"
        )

    if getattr(result, "recurrent_high_rate_detected", False):
        return (
            f"ظهر ارتفاع متكرر في معدل النبض فوق المجال المتوقع، بإجمالي يقارب "
            f"{getattr(result, 'high_rate_total_sec', 0.0):.0f} ثانية فوق 100 ض/د.{bpm_range}"
        )

    if getattr(result, "recurrent_low_rate_detected", False):
        return (
            f"ظهر انخفاض متكرر في معدل النبض تحت المجال المتوقع، بإجمالي يقارب "
            f"{getattr(result, 'low_rate_total_sec', 0.0):.0f} ثانية تحت 60 ض/د.{bpm_range}"
        )

    if avg > 100:
        return f"كان متوسط النبض مرتفعًا نسبيًا، حوالي {avg:.0f} ض/د.{bpm_range}"

    if avg < 60:
        return f"كان متوسط النبض منخفضًا نسبيًا، حوالي {avg:.0f} ض/د.{bpm_range}"

    return f"كان متوسط النبض حوالي {avg:.0f} ض/د، وضمن المجال المتوقع لهذا التسجيل.{bpm_range}"


def _local_english(result: SessionResult, patient_name: str, source_name: str) -> str:
    patient = patient_name or "Unknown"

    if result.n_beats <= 0:
        return (
            f"Patient {patient} was analyzed using {source_name}. No valid beats were available for review. "
            "A new recording with better signal quality is recommended before interpreting the session."
        )

    if result.pct_abnormal >= 40:
        rhythm_note = (
            "The session showed a high abnormal-beat burden, meaning abnormal classifications formed "
            "a large portion of the captured beats."
        )
    elif result.pct_abnormal >= 15:
        rhythm_note = (
            "The session showed a moderate abnormal-beat burden, with intermittent abnormal classifications "
            "during the recording."
        )
    else:
        rhythm_note = "The session was predominantly normal, with most captured beats classified as normal."

    if result.pct_unusable >= 25:
        quality_note = (
            "A considerable portion of the signal was noisy or unusable, so the result should be interpreted cautiously."
        )
    elif result.pct_unusable >= 10:
        quality_note = (
            "Some beats were excluded because of low signal quality, but the session still provides useful trend information."
        )
    else:
        quality_note = "Signal quality was generally acceptable for reviewing the session-level trend."

    rate_note = _rate_note_en(result)

    subtype_note = ""
    if getattr(result, "dominant_arrhythmia_display", ""):
        dist = _mitbih_distribution_text(result)
        subtype_note = (
            f" Additional MIT-BIH subtype review suggested {result.dominant_arrhythmia_display} "
            "as the dominant abnormal subtype among re-analyzed abnormal beats."
        )
        if dist:
            subtype_note += f" Subtype distribution: {dist}."

    recommendation = (
        "This report is intended for software review and screening support only. "
        "It should be interpreted together with patient symptoms, electrode quality, and clinical review."
    )

    return (
        f"Patient {patient} was analyzed using {source_name} over a {int(result.duration_sec)}-second session. "
        f"A total of {result.n_beats} beats were captured: {result.pct_normal:.1f}% normal, "
        f"{result.pct_abnormal:.1f}% abnormal, and {result.pct_unusable:.1f}% unusable/noisy. "
        f"{rhythm_note} {quality_note} {rate_note}{subtype_note} {recommendation}"
    )


def _local_arabic(result: SessionResult, patient_name: str, source_name: str) -> str:
    patient = patient_name or "Unknown"

    if result.n_beats <= 0:
        return (
            f"تم تحليل المريض {patient} باستخدام {source_name}. لم تتوفر نبضات صالحة للمراجعة. "
            "يفضّل إعادة التسجيل بجودة إشارة أفضل قبل تفسير الجلسة."
        )

    if result.pct_abnormal >= 40:
        rhythm_note = (
            "أظهرت الجلسة نسبة مرتفعة من النبضات المصنفة كغير طبيعية، أي أن التصنيفات غير الطبيعية شكّلت جزءًا كبيرًا من النبضات المسجلة."
        )
    elif result.pct_abnormal >= 15:
        rhythm_note = (
            "أظهرت الجلسة نسبة متوسطة من النبضات المصنفة كغير طبيعية، مع ظهور متقطع لهذه النبضات أثناء التسجيل."
        )
    else:
        rhythm_note = "كان النمط العام للجلسة طبيعيًا في الغالب، حيث صُنفت معظم النبضات كطبيعية."

    if result.pct_unusable >= 25:
        quality_note = "جزء ملحوظ من الإشارة كان ضجيجًا أو غير قابل للاستخدام، لذلك يجب تفسير النتيجة بحذر."
    elif result.pct_unusable >= 10:
        quality_note = "تم استبعاد بعض النبضات بسبب انخفاض جودة الإشارة، لكن الجلسة ما زالت تعطي فكرة مفيدة عن الاتجاه العام."
    else:
        quality_note = "كانت جودة الإشارة مناسبة بشكل عام لمراجعة الاتجاه العام للجلسة."

    rate_note = _rate_note_ar(result)

    subtype_note = ""
    if getattr(result, "dominant_arrhythmia_display", ""):
        dist = _mitbih_distribution_text(result)
        subtype_note = (
            f" أظهر تحليل الأنواع الفرعية وفق MIT-BIH أن {result.dominant_arrhythmia_display} "
            "هو النمط غير الطبيعي الأكثر ظهورًا بين النبضات غير الطبيعية المعاد تحليلها."
        )
        if dist:
            subtype_note += f" توزيع الأنواع: {dist}."

    recommendation = (
        "هذا التقرير مخصص للمراجعة البرمجية ودعم الفرز الأولي فقط، ويجب تفسيره مع الأعراض وجودة الأقطاب والمراجعة السريرية."
    )

    return (
        f"تم تحليل المريض {patient} باستخدام {source_name} خلال جلسة مدتها {int(result.duration_sec)} ثانية. "
        f"تم التقاط {result.n_beats} نبضة: {result.pct_normal:.1f}% طبيعية، "
        f"{result.pct_abnormal:.1f}% غير طبيعية، و {result.pct_unusable:.1f}% ضجيج أو غير صالحة. "
        f"{rhythm_note} {quality_note} {rate_note}{subtype_note} {recommendation}"
    )


def generate_bilingual_ai_report(
    result: SessionResult,
    patient_name: str,
    source_name: str,
) -> dict[str, str]:
    """
    Local deterministic report generator.
    No external API is used.
    """
    return {
        "ar": _local_arabic(result, patient_name, source_name),
        "en": _local_english(result, patient_name, source_name),
        "provider": "",
    }