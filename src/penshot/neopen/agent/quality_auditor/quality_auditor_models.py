"""
@FileName: quality_auditor_models.py
@Description: 质量审核模型
@Author: HiPeng
@Github: https://github.com/neopen/story-shot-agent
@Time: 2026/1/19 22:58
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, unique
from typing import List, Optional, Any, Dict, Literal

from pydantic import Field, BaseModel

from penshot.neopen.agent.workflow.workflow_models import PipelineNode


@unique
class SeverityLevel(str, Enum):
    """违规严重程度"""
    INFO = "info"  # 信息级别，不影响执行
    WARNING = "warning"  # 警告级别，建议修复
    MODERATE = "moderate"  # 中度问题，需要调整
    MAJOR = "major"  # 主要问题，需要重新处理
    CRITICAL = "critical"  # 严重问题，需要人工干预
    ERROR = "error"  # 错误级别，无法继续


@unique
class AuditStatus(str, Enum):
    """质量审查状态枚举"""
    PASSED = "passed"  # 完全通过
    MINOR_ISSUES = "minor_issues"  # 轻微问题（警告级别）
    MODERATE_ISSUES = "moderate_issues"  # 中度问题（需要调整）
    MAJOR_ISSUES = "major_issues"  # 主要问题（需要重新处理）
    CRITICAL_ISSUES = "critical_issues"  # 严重问题（需要人工干预）
    FAILED = "failed"  # 完全失败
    NEEDS_HUMAN = "needs_human"  # 需要人工决策

    NEEDS_REVIEW = "needs_review"  # 需要审查
    WARNING = "warning"  # 有警告


class IssueType(str, Enum):
    """问题类型枚举"""
    SCENE = "scene"                 # 场景问题
    CHARACTER = "character"         # 角色问题
    WEATHER = "weather"             # 气候问题
    DIALOGUE = "dialogue"           # 对话问题
    ACTION = "action"               # 动作问题
    DURATION = "duration"           # 时长问题
    PROMPT = "prompt"               # 提示词问题
    AUDIO = "audio"                 # 音频问题
    STYLE = "style"                 # 风格问题
    MODEL = "model"                 # 模型问题
    FRAGMENT = "fragment"           # 片段问题
    SHOT = "shot"                   # 镜头问题
    CONTINUITY = "continuity"       # 连续性问题
    TRUNCATION = "truncation"       # 截断问题
    FORMAT = "format"               # 格式问题
    COMPLETENESS = "completeness"   # 完整性问题
    OTHER = "other"                 # 其他问题


@dataclass
class Rule:
    """规则定义"""
    code: str
    description: str
    issue_type: 'IssueType'


class RuleType(str, Enum):
    """规则类型枚举 - 统一管理所有审查规则"""

    # ==================== LLM审查规则 ====================
    LLM_COHERENCE = Rule("llm_coherence", "LLM连贯性检查", IssueType.CONTINUITY)

    # ==================== 剧本解析规则 (Parse Agent) ====================
    # 场景相关
    SCENE_MISSING = Rule("scene_missing", "场景缺失", IssueType.SCENE)
    SCENE_INSUFFICIENT = Rule("scene_insufficient", "场景数量不足", IssueType.SCENE)
    SCENE_DESC_MISSING = Rule("scene_desc_missing", "场景描述缺失", IssueType.SCENE)

    # 角色相关
    CHARACTER_MISSING = Rule("character_missing", "角色缺失", IssueType.CHARACTER)
    CHARACTER_INCONSISTENT = Rule("character_inconsistent", "角色不一致", IssueType.CHARACTER)
    CHARACTER_DESC_MISSING = Rule("character_desc_missing", "角色描述缺失", IssueType.CHARACTER)
    CHARACTER_DUPLICATE = Rule("character_duplicate", "角色重复", IssueType.CHARACTER)

    # 对话相关
    DIALOGUE_MISSING = Rule("dialogue_missing", "对话缺失", IssueType.DIALOGUE)
    DIALOGUE_INSUFFICIENT = Rule("dialogue_insufficient", "对话提取不足", IssueType.DIALOGUE)
    DIALOGUE_CHARACTER_MISSING = Rule("dialogue_character_missing", "对话缺少角色", IssueType.DIALOGUE)
    DIALOGUE_EMOTION_MISSING = Rule("dialogue_emotion_missing", "对话情感标注缺失", IssueType.DIALOGUE)

    # 动作相关
    ACTION_INSUFFICIENT = Rule("action_insufficient", "动作提取不足", IssueType.ACTION)
    ACTION_DESC_MISSING = Rule("action_desc_missing", "动作描述缺失", IssueType.ACTION)
    ACTION_INTENSITY_DEFAULT = Rule("action_intensity_default", "动作强度使用默认值", IssueType.ACTION)

    # 元素相关
    ELEMENT_DURATION_TOO_SHORT = Rule("element_duration_too_short", "元素时长过短", IssueType.DURATION)
    ELEMENT_DURATION_TOO_LONG = Rule("element_duration_too_long", "元素时长过长", IssueType.DURATION)
    ELEMENT_SEQUENCE_WRONG = Rule("element_sequence_wrong", "元素顺序错误", IssueType.SCENE)
    ELEMENT_ID_FORMAT = Rule("element_id_format", "元素ID格式不规范", IssueType.FORMAT)

    # 情感相关
    EMOTION_MISMATCH = Rule("emotion_mismatch", "情感标注不匹配", IssueType.CHARACTER)
    EMOTION_INCONSISTENT = Rule("emotion_inconsistent", "情感前后不一致", IssueType.CONTINUITY)

    # ==================== 分镜生成规则 (Shot Segmenter) ====================
    # 镜头基本规则
    SHOT_MISSING = Rule("shot_missing", "未能生成任何镜头", IssueType.FRAGMENT)
    SHOT_INSUFFICIENT = Rule("shot_insufficient", "镜头数量不足", IssueType.FRAGMENT)
    SHOT_EXCESSIVE = Rule("shot_excessive", "镜头数量过多", IssueType.FRAGMENT)

    # 镜头时长
    SHOT_DURATION_TOO_SHORT = Rule("shot_duration_too_short", "镜头时长过短", IssueType.DURATION)
    SHOT_DURATION_TOO_LONG = Rule("shot_duration_too_long", "镜头时长过长", IssueType.DURATION)
    SHOT_DURATION_UNBALANCED = Rule("shot_duration_unbalanced", "镜头时长分布不均", IssueType.DURATION)

    # 镜头描述
    SHOT_DESCRIPTION_MISSING = Rule("shot_description_missing", "镜头描述缺失", IssueType.PROMPT)
    SHOT_DESCRIPTION_TOO_SHORT = Rule("shot_description_too_short", "镜头描述过短", IssueType.PROMPT)
    SHOT_DESCRIPTION_VAGUE = Rule("shot_description_vague", "镜头描述模糊", IssueType.PROMPT)

    # 镜头类型
    SHOT_TYPE_UNIFORM = Rule("shot_type_uniform", "镜头类型单一", IssueType.STYLE)
    SHOT_TYPE_INVALID = Rule("shot_type_invalid", "无效的镜头类型", IssueType.STYLE)
    SHOT_REPETITIVE = Rule("shot_repetitive", "镜头类型重复连续", IssueType.CONTINUITY)
    SHOT_TYPE_MISMATCH = Rule("shot_type_mismatch", "镜头类型与内容不匹配", IssueType.STYLE)

    # 角色相关
    CHARACTER_NOT_IN_SHOTS = Rule("character_not_in_shots", "角色未在镜头中出现", IssueType.CHARACTER)
    SHOT_MAIN_CHARACTER_MISSING = Rule("shot_main_character_missing", "镜头缺少主要角色标识", IssueType.CHARACTER)

    # 镜头连续性
    SHOT_TRANSITION_ABRUPT = Rule("shot_transition_abrupt", "镜头切换过于突兀", IssueType.CONTINUITY)
    SHOT_ACTION_BREAK = Rule("shot_action_break", "动作连续性中断", IssueType.CONTINUITY)

    # ==================== 视频分割规则 (Video Splitter) ====================
    # 片段基本规则
    FRAGMENT_MISSING = Rule("fragment_missing", "未生成视频片段", IssueType.FRAGMENT)
    FRAGMENT_INSUFFICIENT = Rule("fragment_insufficient", "片段数量不足", IssueType.FRAGMENT)
    FRAGMENT_EXCESSIVE = Rule("fragment_excessive", "片段数量过多", IssueType.FRAGMENT)

    # 片段时长
    FRAGMENT_DURATION_TOO_SHORT = Rule("fragment_duration_too_short", "片段时长过短", IssueType.DURATION)
    FRAGMENT_DURATION_TOO_LONG = Rule("fragment_duration_too_long", "片段时长过长", IssueType.DURATION)
    FRAGMENT_DURATION_LIMIT = Rule("fragment_duration_limit", "片段时长超出限制", IssueType.DURATION)

    # 片段描述
    FRAGMENT_DESCRIPTION_MISSING = Rule("fragment_description_missing", "片段描述缺失", IssueType.PROMPT)
    FRAGMENT_DESCRIPTION_TOO_SHORT = Rule("fragment_description_too_short", "片段描述过短", IssueType.PROMPT)

    # 片段连续性
    FRAGMENT_TIME_GAP = Rule("fragment_time_gap", "片段间存在时间间隔", IssueType.CONTINUITY)
    FRAGMENT_OVERLAP = Rule("fragment_overlap", "片段间存在时间重叠", IssueType.CONTINUITY)
    FRAGMENT_NO_CONTINUITY = Rule("fragment_no_continuity", "片段缺少连续性注释", IssueType.CONTINUITY)
    FRAGMENT_CONTINUITY_BROKEN = Rule("fragment_continuity_broken", "片段连续性被破坏", IssueType.CONTINUITY)

    # 元素关联
    FRAGMENT_NO_ELEMENTS = Rule("fragment_no_elements", "片段未关联剧本元素", IssueType.FRAGMENT)
    FRAGMENT_ELEMENT_MISMATCH = Rule("fragment_element_mismatch", "片段元素关联错误", IssueType.FRAGMENT)

    # ==================== 提示词转换规则 (Prompt Converter) ====================
    # 提示词基本规则
    PROMPT_MISSING = Rule("prompt_missing", "未生成任何提示词", IssueType.PROMPT)
    PROMPT_EMPTY = Rule("prompt_empty", "提示词为空", IssueType.PROMPT)
    PROMPT_TOO_LONG = Rule("prompt_too_long", "提示词过长", IssueType.PROMPT)
    PROMPT_TOO_SHORT = Rule("prompt_too_short", "提示词过短", IssueType.PROMPT)
    PROMPT_TRUNCATED = Rule("prompt_truncated", "提示词被截断", IssueType.TRUNCATION)
    PROMPT_INCOMPLETE = Rule("prompt_incomplete", "提示词不完整", IssueType.TRUNCATION)

    # 提示词内容
    PROMPT_LOW_QUALITY = Rule("prompt_low_quality", "提示词质量较低", IssueType.PROMPT)
    PROMPT_NO_VISUAL = Rule("prompt_no_visual", "提示词缺少视觉描述", IssueType.PROMPT)
    PROMPT_NO_ACTION = Rule("prompt_no_action", "提示词缺少动作描述", IssueType.ACTION)
    PROMPT_NO_EMOTION = Rule("prompt_no_emotion", "提示词缺少情感描述", IssueType.CHARACTER)

    # 风格相关
    STYLE_INCONSISTENT = Rule("style_inconsistent", "风格不一致", IssueType.STYLE)
    STYLE_INVALID = Rule("style_invalid", "无效的风格", IssueType.STYLE)
    NEGATIVE_PROMPT_MISSING = Rule("negative_prompt_missing", "缺少负面提示词", IssueType.PROMPT)
    NEGATIVE_PROMPT_INSUFFICIENT = Rule("negative_prompt_insufficient", "负面提示词不足", IssueType.PROMPT)

    # 音频相关
    AUDIO_PROMPT_MISSING = Rule("audio_prompt_missing", "缺少音频提示词", IssueType.AUDIO)
    AUDIO_PROMPT_TOO_SHORT = Rule("audio_prompt_too_short", "音频提示词过短", IssueType.AUDIO)
    AUDIO_DURATION_MISMATCH = Rule("audio_duration_mismatch", "音频时长与视频时长不匹配", IssueType.DURATION)
    AUDIO_VOICE_TYPE_INVALID = Rule("audio_voice_type_invalid", "无效的人声类型", IssueType.AUDIO)
    AUDIO_EMOTION_MISMATCH = Rule("audio_emotion_mismatch", "音频情感与画面不匹配", IssueType.AUDIO)

    # 模型相关
    MODEL_UNSUPPORTED = Rule("model_unsupported", "不支持的模型", IssueType.MODEL)
    MODEL_DEPRECATED = Rule("model_deprecated", "模型已弃用", IssueType.MODEL)

    # ==================== 通用规则 ====================
    # 格式规范
    ID_FORMAT_INVALID = Rule("id_format_invalid", "ID格式不规范", IssueType.FORMAT)
    TIMESTAMP_INVALID = Rule("timestamp_invalid", "时间戳无效", IssueType.FORMAT)
    DATA_TYPE_ERROR = Rule("data_type_error", "数据类型错误", IssueType.FORMAT)

    # 一致性检查
    CONSISTENCY_GENERAL = Rule("consistency_general", "一般一致性问题", IssueType.CONTINUITY)
    COLOR_CONSISTENCY = Rule("color_consistency", "色彩风格不一致", IssueType.CONTINUITY)
    LIGHTING_CONSISTENCY = Rule("lighting_consistency", "光照风格不一致", IssueType.CONTINUITY)

    # 完整性检查
    COMPLETENESS_GENERAL = Rule("completeness_general", "内容不完整", IssueType.COMPLETENESS)
    METADATA_MISSING = Rule("metadata_missing", "元数据缺失", IssueType.COMPLETENESS)

    def __init__(self, rule: Rule):
        self._code = rule.code
        self._description = rule.description
        self._issue_type = rule.issue_type

    @property
    def code(self) -> str:
        """获取规则代码"""
        return self._code

    @property
    def description(self) -> str:
        """获取规则描述"""
        return self._description

    @property
    def issue_type(self) -> 'IssueType':
        """获取问题类型"""
        return self._issue_type

    @classmethod
    def from_code(cls, code: str) -> Optional['RuleType']:
        """根据规则代码获取规则"""
        for rule in cls:
            if rule.code == code:
                return rule
        return None

    @classmethod
    def by_issue_type(cls, issue_type: 'IssueType') -> list:
        """根据问题类型获取所有相关规则"""
        return [rule for rule in cls if rule.issue_type == issue_type]

    def __str__(self) -> str:
        return f"{self.code}: {self.description} ({self.issue_type.value})"


class IssueCode(str, Enum):
    """统一的问题编码枚举"""

    # ==================== SCENE 场景问题 ====================
    SCENE_MISSING = "scene_missing"  # 场景缺失
    SCENE_INSUFFICIENT = "scene_insufficient"  # 场景数量不足
    SCENE_DESC_MISSING = "scene_desc_missing"  # 场景描述缺失
    SCENE_LOCATION_INVALID = "scene_location_invalid"  # 场景位置无效

    # ==================== CHARACTER 角色问题 ====================
    CHARACTER_MISSING = "character_missing"  # 角色缺失
    CHARACTER_DESC_MISSING = "character_desc_missing"  # 角色描述缺失
    CHARACTER_INCONSISTENT = "character_inconsistent"  # 角色不一致
    CHARACTER_DUPLICATE = "character_duplicate"  # 角色重复
    CHARACTER_NOT_IN_SHOTS = "character_not_in_shots"  # 角色未在镜头中出现

    # ==================== DIALOGUE 对话问题 ====================
    DIALOGUE_MISSING = "dialogue_missing"  # 对话缺失
    DIALOGUE_CHARACTER_MISSING = "dialogue_character_missing"  # 对话缺少角色
    DIALOGUE_EMOTION_MISSING = "dialogue_emotion_missing"  # 对话情感标注缺失

    # ==================== ACTION 动作问题 ====================
    ACTION_INSUFFICIENT = "action_insufficient"  # 动作提取不足
    ACTION_DESC_MISSING = "action_desc_missing"  # 动作描述缺失
    ACTION_BREAK = "action_break"  # 动作连续性中断

    # ==================== DURATION 时长问题 ====================
    DURATION_TOO_SHORT = "duration_too_short"  # 时长过短
    DURATION_TOO_LONG = "duration_too_long"  # 时长过长
    DURATION_UNBALANCED = "duration_unbalanced"  # 时长分布不均
    DURATION_MISMATCH = "duration_mismatch"  # 时长不匹配

    # ==================== PROMPT 提示词问题 ====================
    PROMPT_MISSING = "prompt_missing"  # 提示词缺失
    PROMPT_EMPTY = "prompt_empty"  # 提示词为空
    PROMPT_TOO_LONG = "prompt_too_long"  # 提示词过长
    PROMPT_TOO_SHORT = "prompt_too_short"  # 提示词过短
    PROMPT_TRUNCATED = "prompt_truncated"  # 提示词被截断
    PROMPT_LOW_QUALITY = "prompt_low_quality"  # 提示词质量较低
    PROMPT_NO_VISUAL = "prompt_no_visual"  # 缺少视觉描述
    PROMPT_NO_ACTION = "prompt_no_action"  # 缺少动作描述
    PROMPT_NO_EMOTION = "prompt_no_emotion"  # 缺少情感描述

    # ==================== NEGATIVE_PROMPT 负面提示词问题 ====================
    NEGATIVE_PROMPT_MISSING = "negative_prompt_missing"  # 缺少负面提示词
    NEGATIVE_PROMPT_INSUFFICIENT = "negative_prompt_insufficient"  # 负面提示词不足

    # ==================== AUDIO 音频问题 ====================
    AUDIO_MISSING = "audio_missing"  # 音频提示词缺失
    AUDIO_TOO_SHORT = "audio_too_short"  # 音频提示词过短
    AUDIO_DURATION_MISMATCH = "audio_duration_mismatch"  # 音频时长不匹配
    AUDIO_VOICE_INVALID = "audio_voice_invalid"  # 人声类型无效

    # ==================== STYLE 风格问题 ====================
    STYLE_INCONSISTENT = "style_inconsistent"  # 风格不一致
    STYLE_INVALID = "style_invalid"  # 无效风格

    # ==================== SHOT 镜头问题 ====================
    SHOT_MISSING = "shot_missing"  # 镜头缺失
    SHOT_INSUFFICIENT = "shot_insufficient"  # 镜头数量不足
    SHOT_EXCESSIVE = "shot_excessive"  # 镜头数量过多
    SHOT_DESC_MISSING = "shot_desc_missing"  # 镜头描述缺失
    SHOT_DESC_TOO_SHORT = "shot_desc_too_short"  # 镜头描述过短
    SHOT_TYPE_UNIFORM = "shot_type_uniform"  # 镜头类型单一
    SHOT_TYPE_INVALID = "shot_type_invalid"  # 镜头类型无效
    SHOT_TYPE_MISMATCH = "shot_type_mismatch"  # 镜头类型与内容不匹配
    SHOT_REPETITIVE = "shot_repetitive"  # 镜头类型重复连续
    SHOT_TRANSITION_ABRUPT = "shot_transition_abrupt"  # 镜头切换突兀
    SHOT_MISSING_CHARACTER = "shot_missing_character"  # 缺少主要角色标识

    # ==================== FRAGMENT 片段问题 ====================
    FRAGMENT_MISSING = "fragment_missing"  # 片段缺失
    FRAGMENT_INSUFFICIENT = "fragment_insufficient"  # 片段数量不足
    FRAGMENT_EXCESSIVE = "fragment_excessive"  # 片段数量过多
    FRAGMENT_DESC_MISSING = "fragment_desc_missing"  # 片段描述缺失
    FRAGMENT_DESC_TOO_SHORT = "fragment_desc_too_short"  # 片段描述过短
    FRAGMENT_OVERLAP = "fragment_overlap"  # 片段时间重叠
    FRAGMENT_GAP = "fragment_gap"  # 片段时间间隔
    FRAGMENT_NO_ELEMENTS = "fragment_no_elements"  # 未关联剧本元素
    FRAGMENT_ELEMENT_MISMATCH = "fragment_element_mismatch"  # 元素关联错误
    FRAGMENT_NO_CONTINUITY = "fragment_no_continuity"  # 缺少连续性注释

    # ==================== CONTINUITY 连续性问题 ====================
    CONTINUITY_BROKEN = "continuity_broken"  # 连续性被破坏
    TIME_GAP = "time_gap"  # 时间间隔
    TIME_OVERLAP = "time_overlap"  # 时间重叠
    EMOTION_INCONSISTENT = "emotion_inconsistent"  # 情感不一致
    COLOR_INCONSISTENT = "color_inconsistent"  # 色彩不一致
    LIGHTING_INCONSISTENT = "lighting_inconsistent"  # 光照不一致

    # ==================== TRUNCATION 截断问题 ====================
    TEXT_TRUNCATED = "text_truncated"  # 文本被截断
    PROMPT_INCOMPLETE = "prompt_incomplete"  # 提示词不完整

    # ==================== FORMAT 格式问题 ====================
    ID_FORMAT_INVALID = "id_format_invalid"  # ID格式无效
    TIMESTAMP_INVALID = "timestamp_invalid"  # 时间戳无效
    DATA_TYPE_ERROR = "data_type_error"  # 数据类型错误

    # ==================== COMPLETENESS 完整性问题 ====================
    CONTENT_INCOMPLETE = "content_incomplete"  # 内容不完整
    METADATA_MISSING = "metadata_missing"  # 元数据缺失

    # ==================== MODEL 模型问题 ====================
    MODEL_UNSUPPORTED = "model_unsupported"  # 不支持的模型
    MODEL_DEPRECATED = "model_deprecated"  # 模型已弃用

    # ==================== WEATHER 气象问题 ====================
    WEATHER_MISSING = "weather_missing"  # 气象信息缺失
    WEATHER_INCONSISTENT = "weather_inconsistent"  # 气象不一致

    # ==================== OTHER 其他问题 ====================
    OTHER = "other"  # 其他问题

    @property
    def category(self) -> str:
        """获取问题类别（issue_type 对应的字符串）"""
        return self.value.split('_')[0] if '_' in self.value else self.value

    @property
    def action(self) -> str:
        """获取修复动作"""
        parts = self.value.split('_')
        return parts[-1] if len(parts) > 1 else 'unknown'

    @classmethod
    def from_rule_type(cls, rule: 'RuleType') -> 'IssueCode':
        """从 RuleType 映射到 IssueCode"""
        mapping = {
            # 场景相关
            RuleType.SCENE_MISSING: cls.SCENE_MISSING,
            RuleType.SCENE_INSUFFICIENT: cls.SCENE_INSUFFICIENT,
            RuleType.SCENE_DESC_MISSING: cls.SCENE_DESC_MISSING,

            # 角色相关
            RuleType.CHARACTER_MISSING: cls.CHARACTER_MISSING,
            RuleType.CHARACTER_INCONSISTENT: cls.CHARACTER_INCONSISTENT,
            RuleType.CHARACTER_DESC_MISSING: cls.CHARACTER_DESC_MISSING,
            RuleType.CHARACTER_DUPLICATE: cls.CHARACTER_DUPLICATE,

            # 对话相关
            RuleType.DIALOGUE_MISSING: cls.DIALOGUE_MISSING,

            # 动作相关
            RuleType.ACTION_INSUFFICIENT: cls.ACTION_INSUFFICIENT,
            RuleType.ACTION_DESC_MISSING: cls.ACTION_DESC_MISSING,

            # 时长相关
            RuleType.SHOT_DURATION_TOO_SHORT: cls.DURATION_TOO_SHORT,
            RuleType.SHOT_DURATION_TOO_LONG: cls.DURATION_TOO_LONG,
            RuleType.FRAGMENT_DURATION_TOO_SHORT: cls.DURATION_TOO_SHORT,
            RuleType.FRAGMENT_DURATION_TOO_LONG: cls.DURATION_TOO_LONG,
            RuleType.AUDIO_DURATION_MISMATCH: cls.DURATION_MISMATCH,

            # 提示词相关
            RuleType.PROMPT_MISSING: cls.PROMPT_MISSING,
            RuleType.PROMPT_EMPTY: cls.PROMPT_EMPTY,
            RuleType.PROMPT_TOO_LONG: cls.PROMPT_TOO_LONG,
            RuleType.PROMPT_TOO_SHORT: cls.PROMPT_TOO_SHORT,
            RuleType.PROMPT_TRUNCATED: cls.PROMPT_TRUNCATED,
            RuleType.PROMPT_LOW_QUALITY: cls.PROMPT_LOW_QUALITY,

            # 风格相关
            RuleType.STYLE_INCONSISTENT: cls.STYLE_INCONSISTENT,

            # 音频相关
            RuleType.AUDIO_PROMPT_MISSING: cls.AUDIO_MISSING,

            # 连续性相关
            RuleType.FRAGMENT_TIME_GAP: cls.TIME_GAP,
            RuleType.FRAGMENT_OVERLAP: cls.TIME_OVERLAP,
            RuleType.SHOT_TRANSITION_ABRUPT: cls.SHOT_TRANSITION_ABRUPT,
            RuleType.SHOT_ACTION_BREAK: cls.ACTION_BREAK,

            # 镜头相关
            RuleType.SHOT_MISSING: cls.SHOT_MISSING,
            RuleType.SHOT_INSUFFICIENT: cls.SHOT_INSUFFICIENT,
            RuleType.SHOT_DESCRIPTION_MISSING: cls.SHOT_DESC_MISSING,
            RuleType.SHOT_TYPE_UNIFORM: cls.SHOT_TYPE_UNIFORM,
            RuleType.SHOT_REPETITIVE: cls.SHOT_REPETITIVE,

            # 片段相关
            RuleType.FRAGMENT_MISSING: cls.FRAGMENT_MISSING,
            RuleType.FRAGMENT_INSUFFICIENT: cls.FRAGMENT_INSUFFICIENT,
            RuleType.FRAGMENT_DESCRIPTION_MISSING: cls.FRAGMENT_DESC_MISSING,
            RuleType.FRAGMENT_NO_ELEMENTS: cls.FRAGMENT_NO_ELEMENTS,

            # 格式相关
            RuleType.ID_FORMAT_INVALID: cls.ID_FORMAT_INVALID,

            # 完整性相关
            RuleType.COMPLETENESS_GENERAL: cls.CONTENT_INCOMPLETE,
        }
        return mapping.get(rule, cls.OTHER)


class BasicViolation(BaseModel):
    """MVP违规记录"""
    issue_type: IssueType = Field(..., description="问题类型")
    issue_code: Optional[str] = Field(
        default=None,
        description="问题子类型，如 'empty', 'too_long', 'missing_character' 等"
    )
    issue_desc: Optional[str] = Field(
        default=None,
        description="问题描述"
    )
    issue_value: Optional[Any] = Field(
        default=None,
        description="问题值，如 '508' 等"
    )
    standard_value: Optional[Any] = Field(
        default=None,
        description="标准规范值"
    )
    source_node: Optional[PipelineNode] = Field(default=None, description="问题来源阶段")
    severity: Literal[SeverityLevel.INFO, SeverityLevel.WARNING, SeverityLevel.ERROR,
    SeverityLevel.MAJOR, SeverityLevel.MODERATE, SeverityLevel.CRITICAL] = Field(
        default=SeverityLevel.WARNING,
        description="严重程度"
    )
    fragment_id: Optional[str] = Field(
        default=None,
        description="涉及的片段ID"
    )
    suggestion: Optional[str] = Field(
        default=None,
        description="改进建议"
    )


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BasicViolation':
        """从字典创建实例"""
        return cls(
            issue_code=data.get("issue_code", ""),
            issue_type=IssueType(data.get("issue_type", "other")),
            issue_desc=data.get("issue_desc", ""),
            severity=SeverityLevel(data.get("severity", "warning")),
            fragment_id=data.get("fragment_id"),
            suggestion=data.get("suggestion"),
            source_node=PipelineNode(data.get("source_node")) if data.get("source_node") else None
        )

    @classmethod
    def create(cls, issue_code: str, issue_type: IssueType, issue_desc: str,
                            issue_value: Optional[Any] = None, standard_value: Optional[Any] = None,
                            severity: SeverityLevel = SeverityLevel.WARNING,
                            fragment_id: Optional[str] = None,
                            suggestion: Optional[str] = None,
                            source_node: PipelineNode = PipelineNode.AUDIT_QUALITY) -> 'BasicViolation':
        """添加违规记录"""
        return BasicViolation(
            issue_type=issue_type,
            issue_code=issue_code,
            issue_desc=issue_desc,
            issue_value=issue_value,
            standard_value=standard_value,
            source_node=source_node,
            severity=severity,
            fragment_id=fragment_id,
            suggestion=suggestion
        )

    @classmethod
    def create_violation(cls, rule_type: RuleType,
                       issue_value: Optional[Any] = None, standard_value: Optional[Any] = None,
                       severity: SeverityLevel = SeverityLevel.WARNING,
                       fragment_id: Optional[str] = None, suggestion: Optional[str] = None,
                       source_node: PipelineNode = PipelineNode.AUDIT_QUALITY) -> 'BasicViolation':
        """添加违规记录"""
        return cls.create(issue_code=rule_type.code, issue_type=rule_type.issue_type, issue_desc=rule_type.description,
                                 issue_value=issue_value, standard_value=standard_value,
                                 severity=severity, fragment_id=fragment_id, suggestion=suggestion, source_node=source_node)


class QualityRepairParams(BaseModel):
    fix_needed: bool = Field(
        default=False,
        description="是否需要修复"
    )

    issue_count: int = Field(
        default=0,
        description="问题数量"
    )

    issue_types: List[str] = Field(
        default_factory=list,
        description="修复类型"
    )

    issues: List[Any] = Field(
        default_factory=list,
        description="完整问题列表（可以是 BasicViolation 或 ContinuityIssue）"
    )

    fragments: List[str] = Field(
        default_factory=list,
        description="对应的片段ID集合"
    )

    suggestions: Dict[str, List[str]] = Field(
        default=None,
        description="修复建议"
    )

    severity_summary: Dict[str, int] = Field(
        default=None,
        description="严重程度摘要"
    )

    def add_issue_type(self, issue_type: Any) -> None:
        """添加问题类型（自动转换为字符串）"""
        if hasattr(issue_type, 'value'):
            type_str = issue_type.value
        else:
            type_str = str(issue_type)

        if type_str not in self.issue_types:
            self.issue_types.append(type_str)

    def add_suggestion(self, key: str, suggestion: str) -> None:
        """添加修复建议"""
        if key not in self.suggestions:
            self.suggestions[key] = []
        self.suggestions[key].append(suggestion)

    def get_issue_type_objects(self) -> List[Any]:
        """获取原始问题类型对象（需要外部转换）"""
        # 注意：这个方法返回的是字符串，需要外部根据上下文转换
        return self.issue_types



class QualityAuditReport(BaseModel):
    """MVP质量审查报告"""

    # 元数据
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "audited_at": datetime.now().isoformat(),
            "version": "mvp_1.0",
            "auditor_type": "basic"
        }
    )

    # 项目信息
    project_info: Dict[str, Any] = Field(
        default_factory=lambda: {
            "title": "",
            "fragment_count": 0,
            "total_duration": 0.0
        }
    )

    # 审查状态
    status: AuditStatus = Field(
        default=AuditStatus.PASSED,
        description="审查状态：passed=通过, needs_revision=需要调整, failed=失败"
    )

    # 检查明细
    checks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="执行的检查项目"
    )

    # 违规记录
    violations: List[BasicViolation] = Field(
        default_factory=list,
        description="发现的违规问题"
    )

    # 统计数据
    stats: Dict[str, Any] = Field(
        default_factory=lambda: {
            "total_checks": 0,
            "passed_checks": 0,
            SeverityLevel.WARNING.value: 0,
            SeverityLevel.ERROR.value: 0,
            "fragments_checked": 0
        }
    )

    # 简单建议
    suggestions: List[str] = Field(
        default_factory=lambda: [
            "检查所有片段时长是否≤5秒",
            "确保没有空提示词"
        ]
    )

    # 最终结论
    conclusion: str = Field(
        default="审查通过，可以开始视频生成",
        description="审查结论"
    )

    score: float = Field(
        default=100.0,
        description="质量评分（0-100）"
    )

    detailed_analysis: Dict[str, Any] = Field(
        default=None,
        description="详细分析报告"
    )

    issues_source: Dict[PipelineNode, List[BasicViolation]] = Field(
        default=None,
        description="问题来源"
    )

    repair_params: Dict[PipelineNode, QualityRepairParams] = Field(
        default=None,
        description="修复参数"
    )

    def calculate_weighted_score(self) -> float:
        """统一计算加权分数"""
        if not self.violations:
            return 100.0

        severity_weights = {
            "info": 0,
            "warning": 5,
            "moderate": 10,
            "major": 25,
            "critical": 40,
            "error": 60,
        }

        base_score = 100.0
        for v in self.violations:
            severity = v.severity
            if hasattr(severity, 'value'):
                severity = severity.value
            penalty = severity_weights.get(str(severity).lower(), 5)
            base_score -= penalty

        return max(0.0, min(100.0, base_score))

    def determine_status_from_score(self) -> 'AuditStatus':
        """根据分数确定状态"""
        if self.score >= 90:
            return AuditStatus.PASSED
        elif self.score >= 75:
            return AuditStatus.MINOR_ISSUES
        elif self.score >= 60:
            return AuditStatus.MODERATE_ISSUES
        elif self.score >= 40:
            return AuditStatus.MAJOR_ISSUES
        elif self.score >= 20:
            return AuditStatus.CRITICAL_ISSUES
        else:
            return AuditStatus.FAILED

    def _has_severity(self, severity: SeverityLevel) -> bool:
        """检查是否存在指定严重程度的问题"""
        if not hasattr(self, 'stats'):
            return False
        severity_str = severity.value if hasattr(severity, 'value') else str(severity)
        return self.stats.get(severity_str, 0) > 0


@dataclass
class RepairHistory:
    """修复历史记录"""
    timestamp: float
    stage: str
    issue_count: int
    issues_fixed: List[str]
    success: bool